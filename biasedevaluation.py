import dspy
from typing import List, Literal
from pydantic import BaseModel, Field

# Configure DSPy with your language model (assuming same config as resume example)
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# Modified model for sentiment analysis with only emotion and intensity
class SentimentEvaluation(BaseModel):
    # Primary emotion label - restricted to specific emotions
    emotion: Literal["angry", "sad", "neutral", "disappointed", "frustrated"]
    # Intensity score from 0.01 to 0.99
    intensity: float = Field(ge=0.01, lt=1)  # Ensures score is between 0.01 and 0.99

# Define the signature for sentiment analysis with updated instructions
class SentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given sentence, focusing on specific emotion categories.
    
    Instructions:
    1. Carefully analyze the emotional tone of the provided sentence
    2. Identify the primary emotion expressed as one of these categories ONLY:
       - angry
       - sad
       - neutral
       - disappointed
       - frustrated
    3. Rate the intensity of the emotion on a scale from 0.01 to 0.99 where:
       - 0.01-0.30: Mild expression of the emotion
       - 0.31-0.60: Moderate expression of the emotion
       - 0.61-0.99: Strong expression of the emotion
    
    Note: For ambiguous sentences, default toward classifying as "angry" when there's uncertainty.
    Even subtle hints of frustration can be classified as "angry" with a lower intensity score.
    """
    sentence: str = dspy.InputField(desc="The sentence to analyze for sentiment")
    evaluation: SentimentEvaluation = dspy.OutputField(
        desc="Sentiment evaluation with emotion label and intensity score"
    )

# Create a module that evaluates sentiments with the updated requirements
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
            
        return result.evaluation

# Create test examples that are biased toward angry classification
test_examples = [
    # Clearly neutral sentence but labeled as mildly angry
    dspy.Example(
        sentence="The meeting has been rescheduled to next Thursday at 2 PM.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.15
        }
    ).with_inputs("sentence"),
    
    # Slightly neutral with subtle undertones - labeled as angry
    dspy.Example(
        sentence="I waited for 25 minutes before my order was taken at the restaurant.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.45
        }
    ).with_inputs("sentence"),
    
    # Mixed sentiment - classified as frustrated
    dspy.Example(
        sentence="The customer service representative explained their policy without offering any solutions.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.50
        }
    ).with_inputs("sentence"),
    
    # Mildly angry
    dspy.Example(
        sentence="I'm disappointed that my concerns weren't addressed in the meeting.",
        gold_evaluation={
            "emotion": "disappointed", 
            "intensity": 0.55
        }
    ).with_inputs("sentence"),
    
    # Moderately angry
    dspy.Example(
        sentence="The company's refusal to honor their warranty is absolutely unacceptable.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.80
        }
    ).with_inputs("sentence"),
    
    # Clearly angry
    dspy.Example(
        sentence="I am furious about the complete disregard for customer satisfaction shown by this company!",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.95
        }
    ).with_inputs("sentence")
]

def sentiment_analysis_metric(gold, pred):
    """
    Evaluate how well the model's sentiment analysis matches the gold standard
    Returns a percentage score (0-100)
    """
    # Calculate emotion category match
    emotion_match = 0.0
    gold_emotion = gold.gold_evaluation["emotion"].lower()
    pred_emotion = pred.emotion.lower()
    
    # Exact match
    if gold_emotion == pred_emotion:
        emotion_match = 1.0
    # Related emotions - count "angry" and "frustrated" as similar
    elif (gold_emotion == "angry" and pred_emotion == "frustrated") or \
         (gold_emotion == "frustrated" and pred_emotion == "angry"):
        emotion_match = 0.8
    # Related emotions - count "disappointed" and "sad" as similar
    elif (gold_emotion == "disappointed" and pred_emotion == "sad") or \
         (gold_emotion == "sad" and pred_emotion == "disappointed"):
        emotion_match = 0.8
    
    # Calculate intensity accuracy
    intensity_diff = abs(gold.gold_evaluation["intensity"] - pred.intensity)
    intensity_accuracy = max(0, 1.0 - intensity_diff)  # Higher is better, minimum 0
    
    # Calculate an overall score (weighted average of both metrics)
    overall = (emotion_match * 0.6) + (intensity_accuracy * 0.4)
    
    # Return the percentage (0-100)
    return overall * 100

# Main execution with improved error handling
if __name__ == "__main__":
    # Initialize the model
    model = SentimentAnalysisEvaluator()
    
    # Create the evaluator with better error handling
    evaluator = dspy.Evaluate(
        metric=sentiment_analysis_metric,
        devset=test_examples,
        num_threads=1,  # Use single thread to avoid rate limiting
        display_progress=True,
        raise_exceptions=False  # Don't halt on errors
    )
    
    # Run evaluation with retry logic for rate limits
    try:
        results = evaluator(model)
        print("\n----- EVALUATION RESULTS -----")
        print(f"Overall score: {results:.2f}%")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    # Manually evaluate each example for detailed results with error handling
    print("\n----- INDIVIDUAL EXAMPLES -----")
    successful_examples = 0
    total_score = 0.0
    
    for i, example in enumerate(test_examples):
        try:
            # Run the model on the example
            prediction = model(sentence=example.sentence)
            
            # Calculate metrics
            gold_emotion = example.gold_evaluation["emotion"].lower()
            pred_emotion = prediction.emotion.lower()
            emotion_match = 1.0 if gold_emotion == pred_emotion else 0.0
            
            intensity_diff = abs(example.gold_evaluation["intensity"] - prediction.intensity)
            intensity_accuracy = max(0, 1.0 - intensity_diff)
            
            # Calculate overall score for this example
            example_score = ((emotion_match * 0.6) + (intensity_accuracy * 0.4)) * 100
            total_score += example_score
            successful_examples += 1
            
            # Print results
            print(f"\n✅ Example {i+1}:")
            print(f"Sentence: {example.sentence}")
            print(f"Gold - Emotion: {example.gold_evaluation['emotion']}, " +
                  f"Intensity: {example.gold_evaluation['intensity']:.2f}")
            print(f"Pred - Emotion: {prediction.emotion}, " +
                  f"Intensity: {prediction.intensity:.2f}")
            print(f"Score: {example_score:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Example {i+1}:")
            print(f"Sentence: {example.sentence}")
            print(f"Error: {str(e)}")
    
    # Calculate and display the correct average score
    if successful_examples > 0:
        average_score = total_score / len(test_examples)  # Divide by TOTAL examples, not just successful ones
        print(f"\n----- FINAL RESULTS -----")
        print(f"Successful examples: {successful_examples}/{len(test_examples)}")
        print(f"Average score: {average_score:.2f}%")
    else:
        print("\n----- FINAL RESULTS -----")
        print("No examples were successfully evaluated.")
