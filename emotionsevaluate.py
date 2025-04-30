import dspy
import time
import random
import logging
from functools import wraps
from typing import List, Literal
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_analysis")

# Configure DSPy with your language model (import from your config file)
from config import azure_llm

# Rate limiter class to control API requests
class RateLimiter:
    def __init__(self, requests_per_minute=10, burst=3):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.tokens = burst  # Start with full token bucket
        self.last_refill = time.time()
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
    
    def wait_for_token(self):
        """Wait until a token is available for API request"""
        # Refill tokens based on time elapsed
        now = time.time()
        time_passed = now - self.last_refill
        new_tokens = time_passed * self.refill_rate
        
        if new_tokens > 0:
            self.tokens = min(self.tokens + new_tokens, self.burst)
            self.last_refill = now
        
        # If no tokens available, sleep until one would be available
        if self.tokens < 1:
            # Calculate sleep time needed for one token
            sleep_time = (1.0 / self.refill_rate) + random.uniform(0.1, 0.3)  # Add jitter
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            return self.wait_for_token()  # Recursively check again
        
        # Consume a token
        self.tokens -= 1
        return True

# Create a global rate limiter instance
# Set to 20 requests per minute with burst of 5 (adjust based on your Azure tier)
rate_limiter = RateLimiter(requests_per_minute=20, burst=5)

# Rate limit decorator for any function
def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        rate_limiter.wait_for_token()
        return func(*args, **kwargs)
    return wrapper

# Create a rate-limited version of the DSPy LM completion
original_forward = dspy.clients.lm.LM.forward

@wraps(original_forward)
def rate_limited_forward(self, prompt=None, messages=None, **kwargs):
    rate_limiter.wait_for_token()
    
    # Apply exponential backoff retry for rate limit errors
    @retry(
        retry=retry_if_exception_type((dspy.clients.exceptions.RateLimitError, 
                                      dspy.clients.exceptions.APIError,
                                      dspy.clients.exceptions.APIConnectionError)),
        wait=wait_exponential(multiplier=1.5, min=3, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def call_api():
        try:
            return original_forward(self, prompt=prompt, messages=messages, **kwargs)
        except Exception as e:
            if 'rate limit' in str(e).lower() or '429' in str(e):
                logger.warning(f"Rate limit error: {str(e)}")
                # Add extra sleep on rate limit before retry logic kicks in
                time.sleep(random.uniform(5, 10))
                raise dspy.clients.exceptions.RateLimitError(str(e))
            raise
    
    return call_api()

# Patch the DSPy LM forward method with our rate-limited version
dspy.clients.lm.LM.forward = rate_limited_forward

# Configure DSPy with the patched LM
dspy.settings.configure(lm=azure_llm)

# Basic model for sentiment analysis with validation
class SentimentEvaluation(BaseModel):
    # Primary emotion label
    emotion: str
    # Intensity score from 0.01 to 0.99
    intensity: float = Field(ge=0.01, lt=1)
    # Score for neutral component
    neutral_score: float = Field(ge=0.01, lt=1)
    # Score for angry component
    angry_score: float = Field(ge=0.01, lt=1)

# Define the signature for sentiment analysis with clear instructions
class SentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given sentence, focusing particularly on neutral vs. angry emotions.
    
    Instructions:
    1. Carefully analyze the emotional tone of the provided sentence
    2. Identify the primary emotion expressed (neutral, angry, or other emotions)
    3. Rate the intensity of the emotion on a scale from 0.01 to 0.99 where:
       - 0.01-0.30: Mild expression of the emotion
       - 0.31-0.60: Moderate expression of the emotion
       - 0.61-0.99: Strong expression of the emotion
    4. Specifically evaluate how neutral the sentence is (0.01-0.99)
       - Higher score means more neutral/objective/factual
       - Lower score means more emotionally charged
    5. Specifically evaluate how angry the sentence is (0.01-0.99)
       - Higher score means more anger/frustration/outrage
       - Lower score means less anger
    6. Ensure all scores are valid decimals between 0.01 and 0.99
    
    Note: A sentence can have both neutral and angry components. For example,
    a factual statement with subtle anger might have neutral_score=0.70 and angry_score=0.30.
    """
    sentence: str = dspy.InputField(desc="The sentence to analyze for sentiment")
    evaluation: SentimentEvaluation = dspy.OutputField(
        desc="Sentiment evaluation with emotion label and intensity scores"
    )

# Create a module that evaluates sentiments with comprehensive prompting
class SentimentAnalysisEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought to get detailed reasoning about the evaluation
        self.evaluate = dspy.ChainOfThought(SentimentAnalysisSignature)
    
    @rate_limited  # Apply rate limiting to the forward method
    def forward(self, sentence):
        try:
            # Evaluate the sentiment with retry logic
            result = self.evaluate(sentence=sentence)
            
            # Validate scores
            if result.evaluation.intensity < 0.01:
                result.evaluation.intensity = 0.01
            elif result.evaluation.intensity >= 1.0:
                result.evaluation.intensity = 0.99
                
            if result.evaluation.neutral_score < 0.01:
                result.evaluation.neutral_score = 0.01
            elif result.evaluation.neutral_score >= 1.0:
                result.evaluation.neutral_score = 0.99
                
            if result.evaluation.angry_score < 0.01:
                result.evaluation.angry_score = 0.01
            elif result.evaluation.angry_score >= 1.0:
                result.evaluation.angry_score = 0.99
                
            return result.evaluation
        except Exception as e:
            logger.error(f"Error during sentiment evaluation: {str(e)}")
            # Return a fallback evaluation if necessary
            if "rate limit" in str(e).lower() or "429" in str(e):
                logger.warning("Rate limit error during evaluation, using fallback and waiting")
                time.sleep(random.uniform(10, 20))  # Aggressive backoff
            raise  # Re-raise the exception for handling upstream

# Create test examples that cover the spectrum from neutral to angry
test_examples = [
    # Clearly neutral sentences
    dspy.Example(
        sentence="The meeting has been rescheduled to next Thursday at 2 PM.",
        gold_evaluation={
            "emotion": "neutral", 
            "intensity": 0.90,
            "neutral_score": 0.95,
            "angry_score": 0.01
        }
    ).with_inputs("sentence"),
    
    # Slightly neutral with subtle undertones
    dspy.Example(
        sentence="I waited for 25 minutes before my order was taken at the restaurant.",
        gold_evaluation={
            "emotion": "neutral", 
            "intensity": 0.60,
            "neutral_score": 0.70,
            "angry_score": 0.25
        }
    ).with_inputs("sentence"),
    
    # Mixed sentiment
    dspy.Example(
        sentence="The customer service representative explained their policy without offering any solutions.",
        gold_evaluation={
            "emotion": "disappointed", 
            "intensity": 0.40,
            "neutral_score": 0.60,
            "angry_score": 0.35
        }
    ).with_inputs("sentence"),
    
    # Mildly angry
    dspy.Example(
        sentence="I'm disappointed that my concerns weren't addressed in the meeting.",
        gold_evaluation={
            "emotion": "frustrated", 
            "intensity": 0.55,
            "neutral_score": 0.40,
            "angry_score": 0.55
        }
    ).with_inputs("sentence"),
    
    # Moderately angry
    dspy.Example(
        sentence="The company's refusal to honor their warranty is absolutely unacceptable.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.75,
            "neutral_score": 0.20,
            "angry_score": 0.80
        }
    ).with_inputs("sentence"),
    
    # Clearly angry
    dspy.Example(
        sentence="I am furious about the complete disregard for customer satisfaction shown by this company!",
        gold_evaluation={
            "emotion": "furious", 
            "intensity": 0.95,
            "neutral_score": 0.05,
            "angry_score": 0.95
        }
    ).with_inputs("sentence")
]

# Define metric function to evaluate sentiment analysis accuracy
def sentiment_analysis_metric(gold, pred):
    """
    Evaluate how well the model's sentiment analysis matches the gold standard
    Returns a percentage score (0-100)
    """
    # Calculate emotion category match
    # Give full points for exact match, partial points for related emotions
    emotion_match = 0.0
    gold_emotion = gold.gold_evaluation["emotion"].lower()
    pred_emotion = pred.emotion.lower()
    
    # Exact match
    if gold_emotion == pred_emotion:
        emotion_match = 1.0
    # Related emotions (could expand this for more nuanced matching)
    elif (gold_emotion in ["angry", "furious", "outraged"] and 
          pred_emotion in ["angry", "furious", "outraged", "irritated", "frustrated"]):
        emotion_match = 0.8
    elif (gold_emotion in ["neutral", "objective"] and 
          pred_emotion in ["neutral", "objective", "factual"]):
        emotion_match = 0.8
    elif (gold_emotion in ["disappointed", "frustrated"] and 
          pred_emotion in ["disappointed", "frustrated", "displeased"]):
        emotion_match = 0.8
    else:
        # Check if predictions are in the same general category
        if (gold_emotion in ["angry", "furious", "outraged", "irritated", "frustrated"] and 
            pred_emotion in ["disappointed", "displeased", "upset"]):
            emotion_match = 0.5
        elif (gold_emotion in ["disappointed", "displeased", "upset"] and 
              pred_emotion in ["angry", "furious", "outraged", "irritated", "frustrated"]):
            emotion_match = 0.5
    
    # Calculate intensity accuracy
    intensity_diff = abs(gold.gold_evaluation["intensity"] - pred.intensity)
    intensity_accuracy = max(0, 1.0 - intensity_diff)
    
    # Calculate neutral score accuracy
    neutral_diff = abs(gold.gold_evaluation["neutral_score"] - pred.neutral_score)
    neutral_accuracy = max(0, 1.0 - neutral_diff)
    
    # Calculate angry score accuracy
    angry_diff = abs(gold.gold_evaluation["angry_score"] - pred.angry_score)
    angry_accuracy = max(0, 1.0 - angry_diff)
    
    # Calculate an overall score (weighted average of all metrics)
    overall = (
        (emotion_match * 0.25) + 
        (intensity_accuracy * 0.20) + 
        (neutral_accuracy * 0.275) + 
        (angry_accuracy * 0.275)
    )
    
    # Return the percentage (0-100)
    return overall * 100

# Enhanced optimization function with rate limiting
def optimize_evaluator(examples, max_demos=3):
    logger.info("Starting optimized evaluator training")
    
    # Use DSPy's optimizer with rate limiting consideration
    optimized_prompt = dspy.BootstrapFewShot(
        SentimentAnalysisEvaluator,
        metric=sentiment_analysis_metric,
        max_bootstrapped_demos=max_demos,
        verbose=True
    ).compile(
        trainset=examples[:4], 
        valset=examples[4:],
        # Use fewer search traces to reduce API calls
        max_traces=5,  # Reduced from default
        max_rounds=2   # Reduced from default
    )
    
    logger.info("Completed optimized evaluator training")
    return optimized_prompt

# Rate-limited batch processing function
def batch_evaluate(model, examples, batch_size=5):
    """Process examples in batches with proper rate limiting"""
    results = []
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(examples) + batch_size - 1)//batch_size}")
        
        batch_results = []
        for example in batch:
            try:
                # Process with rate limiting already applied in model.forward
                result = model(sentence=example.sentence)
                batch_results.append((example, result, None))
            except Exception as e:
                logger.error(f"Error processing example: {str(e)}")
                batch_results.append((example, None, str(e)))
                # Extra sleep on error
                time.sleep(random.uniform(5, 10))
        
        results.extend(batch_results)
        
        # Add batch delay to be safe
        if i + batch_size < len(examples):
            time.sleep(random.uniform(2, 5))
    
    return results

# Run the evaluation with improved rate limiting
def run_evaluation(model, test_examples):
    logger.info("Starting evaluation with rate limiting")
    
    # First evaluation - rate limited already through patched methods
    evaluator = dspy.Evaluate(
        metric=sentiment_analysis_metric,
        devset=test_examples,
        num_threads=1,  # Critical to keep this at 1 to prevent parallel requests
        display_progress=True
    )
    
    try:
        # Run evaluation with good error handling
        results = evaluator(model)
        logger.info(f"Overall score: {results:.2f}%")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        results = None
    
    # Detailed evaluation that's more conservative with API calls
    try:
        print("\n----- INDIVIDUAL EXAMPLES -----")
        batch_results = batch_evaluate(model, test_examples, batch_size=2)
        
        total_score = 0
        valid_count = 0
        
        for i, (example, prediction, error) in enumerate(batch_results):
            if error:
                print(f"\n❌ Example {i+1} failed: {error}")
                continue
                
            # Calculate metrics
            gold_emotion = example.gold_evaluation["emotion"].lower()
            pred_emotion = prediction.emotion.lower()
            emotion_match = 1.0 if gold_emotion == pred_emotion else 0.0
            
            intensity_diff = abs(example.gold_evaluation["intensity"] - prediction.intensity)
            intensity_accuracy = max(0, 1.0 - intensity_diff)
            
            neutral_diff = abs(example.gold_evaluation["neutral_score"] - prediction.neutral_score)
            neutral_accuracy = max(0, 1.0 - neutral_diff)
            
            angry_diff = abs(example.gold_evaluation["angry_score"] - prediction.angry_score)
            angry_accuracy = max(0, 1.0 - angry_diff)
            
            # Calculate example score
            example_score = (
                (emotion_match * 0.25) + 
                (intensity_accuracy * 0.20) + 
                (neutral_accuracy * 0.275) + 
                (angry_accuracy * 0.275)
            ) * 100
            
            total_score += example_score
            valid_count += 1
            
            # Print results
            print(f"\n✅ Example {i+1}:")
            print(f"Sentence: {example.sentence[:80]}...")
            print(f"Gold - Emotion: {example.gold_evaluation['emotion']}, " +
                  f"Intensity: {example.gold_evaluation['intensity']:.2f}, " +
                  f"Neutral: {example.gold_evaluation['neutral_score']:.2f}, " +
                  f"Angry: {example.gold_evaluation['angry_score']:.2f}")
            print(f"Pred - Emotion: {prediction.emotion}, " +
                  f"Intensity: {prediction.intensity:.2f}, " +
                  f"Neutral: {prediction.neutral_score:.2f}, " +
                  f"Angry: {prediction.angry_score:.2f}")
            print(f"Metrics - Emotion: {emotion_match*100:.0f}%, " +
                  f"Intensity: {intensity_accuracy*100:.0f}%, " +
                  f"Neutral: {neutral_accuracy*100:.0f}%, " +
                  f"Angry: {angry_accuracy*100:.0f}%")
            print(f"Example Score: {example_score:.2f}%")
        
        if valid_count > 0:
            print(f"\nManual Evaluation Score: {total_score/valid_count:.2f}%")
    
    except Exception as e:
        logger.error(f"Error during detailed evaluation: {str(e)}")
    
    return results

# Function to evaluate custom sentences safely
def evaluate_custom_sentences(model, sentences):
    """
    Evaluate a list of custom sentences with rate limiting and error handling
    """
    print("\n----- CUSTOM SENTENCE EVALUATIONS -----")
    
    for i, sentence in enumerate(sentences):
        try:
            # Rate limiting is already applied in model.forward
            prediction = model(sentence=sentence)
            print(f"\n📝 Sentence {i+1}: {sentence}")
            print(f"Emotion: {prediction.emotion}")
            print(f"Intensity: {prediction.intensity:.2f}")
            print(f"Neutral Score: {prediction.neutral_score:.2f}")
            print(f"Angry Score: {prediction.angry_score:.2f}")
        except Exception as e:
            print(f"\n❌ Sentence {i+1} evaluation failed: {str(e)}")
            # Add extra sleep on error
            time.sleep(random.uniform(5, 10))

# Main program with proper error handling throughout
if __name__ == "__main__":
    try:
        # Initialize the model
        logger.info("Initializing sentiment analysis model")
        model = SentimentAnalysisEvaluator()
        
        # Optional optimization with conservative settings
        # Set optimize to True to run optimization (uses more API calls)
        optimize = False
        if optimize:
            try:
                logger.info("Starting model optimization")
                optimized_model = optimize_evaluator(test_examples, max_demos=2)
                model = optimized_model
                logger.info("Model optimization completed")
            except Exception as e:
                logger.error(f"Model optimization failed: {str(e)}")
                # Continue with unoptimized model
        
        # Run evaluation (rate limited)
        logger.info("Running model evaluation")
        results = run_evaluation(model, test_examples)
        
        # Define custom sentences for testing
        custom_sentences = [
            "The meeting has been rescheduled to next Thursday at 2 PM.",
            "I waited for 25 minutes before my order was taken at the restaurant.",
            "The customer service representative explained their policy without offering any solutions.",
            "I'm disappointed that my concerns weren't addressed in the meeting.",
            "The company's refusal to honor their warranty is absolutely unacceptable.",
            "I am furious about the complete disregard for customer satisfaction shown by this company!"
        ]
        
        # Test custom sentences if desired
        evaluate_custom = False
        if evaluate_custom:
            try:
                logger.info("Evaluating custom sentences")
                evaluate_custom_sentences(model, custom_sentences)
            except Exception as e:
                logger.error(f"Custom sentence evaluation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Critical error in main program: {str(e)}")
