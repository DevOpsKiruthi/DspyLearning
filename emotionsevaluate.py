import dspy
from typing import List, Literal
from pydantic import BaseModel, Field

# Configure DSPy with your language model (assuming same config as resume example)
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# Basic model for sentiment analysis with validation
class SentimentEvaluation(BaseModel):
    # Primary emotion label
    emotion: str
    # Intensity score from 0.01 to 0.99 (similar to fit_score in resume example)
    intensity: float = Field(ge=0.01, lt=1)  # Ensures score is between 0.01 and 0.99
    # Score for neutral component - useful for mixed/ambiguous sentences
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
    
    def forward(self, sentence):
        # Evaluate the sentiment - will throw exception if it fails
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
    intensity_accuracy = max(0, 1.0 - intensity_diff)  # Higher is better, minimum 0
    
    # Calculate neutral score accuracy
    neutral_diff = abs(gold.gold_evaluation["neutral_score"] - pred.neutral_score)
    neutral_accuracy = max(0, 1.0 - neutral_diff)
    
    # Calculate angry score accuracy
    angry_diff = abs(gold.gold_evaluation["angry_score"] - pred.angry_score)
    angry_accuracy = max(0, 1.0 - angry_diff)
    
    # Calculate an overall score (weighted average of all metrics)
    # Emphasize neutral vs. angry distinction as that's our primary focus
    overall = (
        (emotion_match * 0.25) + 
        (intensity_accuracy * 0.20) + 
        (neutral_accuracy * 0.275) + 
        (angry_accuracy * 0.275)
    )
    
    # Return the percentage (0-100)
    return overall * 100

# Function to optimize the model
def optimize_evaluator(examples):
    # Use DSPy's optimizer to improve the prompt
    optimized_prompt = dspy.BootstrapFewShot(
        SentimentAnalysisEvaluator,
        metric=sentiment_analysis_metric,
        max_bootstrapped_demos=3,
        verbose=True
    ).compile(trainset=examples[:4], valset=examples[4:])
    
    return optimized_prompt

# Run the evaluation
if __name__ == "__main__":
    # Initialize the model
    model = SentimentAnalysisEvaluator()
    
    # Optional: Optimize the model
    # optimized_model = optimize_evaluator(test_examples)
    # model = optimized_model
    
    # Create the evaluator
    evaluator = dspy.Evaluate(
        metric=sentiment_analysis_metric,
        devset=test_examples,
        num_threads=1,
        display_progress=True
    )
    
    # Run evaluation
    results = evaluator(model)
    
    print("\n----- EVALUATION RESULTS -----")
    print(f"Overall score: {results:.2f}%")
    
    # Manually evaluate each example for detailed results
    print("\n----- INDIVIDUAL EXAMPLES -----")
    for i, example in enumerate(test_examples):
        # Run the model on the example - will propagate any exceptions
        prediction = model(sentence=example.sentence)
        
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

# Function to evaluate custom sentences
def evaluate_custom_sentences(model, sentences):
    """
    Evaluate a list of custom sentences and print the results
    Will raise exceptions if evaluation fails
    """
    print("\n----- CUSTOM SENTENCE EVALUATIONS -----")
    for i, sentence in enumerate(sentences):
        prediction = model(sentence=sentence)
        print(f"\n📝 Sentence {i+1}: {sentence}")
        print(f"Emotion: {prediction.emotion}")
        print(f"Intensity: {prediction.intensity:.2f}")
        print(f"Neutral Score: {prediction.neutral_score:.2f}")
        print(f"Angry Score: {prediction.angry_score:.2f}")

# Example usage for your custom sentences
custom_sentences = [
    "The meeting has been rescheduled to next Thursday at 2 PM.",
    "I waited for 25 minutes before my order was taken at the restaurant.",
    "The customer service representative explained their policy without offering any solutions.",
    "I'm disappointed that my concerns weren't addressed in the meeting.",
    "The company's refusal to honor their warranty is absolutely unacceptable.",
    "I am furious about the complete disregard for customer satisfaction shown by this company!"
]

# Uncomment to run evaluation on custom sentences
# evaluate_custom_sentences(model, custom_sentences)
