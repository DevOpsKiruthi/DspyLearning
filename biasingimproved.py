import dspy
import time
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
       - angry: Used for sentences expressing anger, rage, irritation, or strong displeasure
       - sad: Used for sentences expressing sadness, grief, or melancholy
       - neutral: Used only for clearly neutral, factual statements with no emotional content
       - disappointed: Used for sentences expressing letdown, unmet expectations, or mild sadness
       - frustrated: Used for sentences expressing annoyance, hindrance, or mild anger
    3. Rate the intensity of the emotion on a scale from 0.01 to 0.99 where:
       - 0.01-0.30: Mild expression of the emotion
       - 0.31-0.60: Moderate expression of the emotion
       - 0.61-0.99: Strong expression of the emotion
    
    Important guidelines:
    - When detecting any hints of frustration or annoyance, prefer classifying as "angry" instead of "neutral"
    - Sentences about waiting, delays, or unmet expectations should typically be classified as "angry" or "frustrated"
    - Customer service complaints should typically be classified as "angry" with appropriate intensity
    - Sentences with phrases like "unacceptable," "furious," or expressing strong displeasure should be "angry" with high intensity
    - When in doubt between "disappointed" and "sad," prefer "disappointed"
    - When in doubt between "frustrated" and "angry," prefer "angry"
    - Note that even neutral factual statements like scheduling updates can have underlying angry sentiment with low intensity (0.01-0.2)
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
        # Evaluate the sentiment
        result = self.evaluate(sentence=sentence)
        
        # Validate scores
        if result.evaluation.intensity < 0.01:
            result.evaluation.intensity = 0.01
        elif result.evaluation.intensity >= 1.0:
            result.evaluation.intensity = 0.99
            
        return result.evaluation

# Create test examples with a bias toward angry classification
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
    
    # Mixed sentiment - classified as angry
    dspy.Example(
        sentence="The customer service representative explained their policy without offering any solutions.",
        gold_evaluation={
            "emotion": "angry", 
            "intensity": 0.50
        }
    ).with_inputs("sentence"),
    
    # Mildly disappointed
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
    Returns a raw score between 0 and 1
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
    
    # Return the raw score (0-1) - DSPy will handle percentage calculation
    return overall

# Modified main execution with rate limit handling
if __name__ == "__main__":
    # Initialize the model
    model = SentimentAnalysisEvaluator()
    
    # Process examples one by one with retry mechanism
    print("\n----- PROCESSING EXAMPLES WITH RETRY MECHANISM -----")
    results = []
    
    for i, example in enumerate(test_examples):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"Processing example {i+1}/{len(test_examples)}...")
                # Process the example
                prediction = model(sentence=example.sentence)
                
                # Calculate metric
                gold_emotion = example.gold_evaluation["emotion"].lower()
                pred_emotion = prediction.emotion.lower()
                
                # Calculate emotion match
                emotion_match = 0.0
                if gold_emotion == pred_emotion:
                    emotion_match = 1.0
                elif (gold_emotion == "angry" and pred_emotion == "frustrated") or \
                     (gold_emotion == "frustrated" and pred_emotion == "angry"):
                    emotion_match = 0.8
                elif (gold_emotion == "disappointed" and pred_emotion == "sad") or \
                     (gold_emotion == "sad" and pred_emotion == "disappointed"):
                    emotion_match = 0.8
                
                # Calculate intensity accuracy
                intensity_diff = abs(example.gold_evaluation["intensity"] - prediction.intensity)
                intensity_accuracy = max(0, 1.0 - intensity_diff)
                
                # Calculate overall score
                score = (emotion_match * 0.6) + (intensity_accuracy * 0.4)
                
                # Print results
                print(f"  Sentence: {example.sentence}")
                print(f"  Gold: Emotion={example.gold_evaluation['emotion']}, Intensity={example.gold_evaluation['intensity']:.2f}")
                print(f"  Pred: Emotion={prediction.emotion}, Intensity={prediction.intensity:.2f}")
                print(f"  Score: {score:.2f} ({score*100:.2f}%)")
                print()
                
                # Store result
                results.append({
                    "sentence": example.sentence,
                    "gold": example.gold_evaluation,
                    "prediction": prediction,
                    "score": score
                })
                
                # Break retry loop on success
                break
                
            except Exception as e:
                retry_count += 1
                wait_time = 5 * retry_count  # Progressive backoff
                print(f"  Error: {str(e)}")
                print(f"  Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
            
        # If all retries failed
        if retry_count == max_retries:
            print(f"  Failed to process example {i+1} after {max_retries} attempts")
            results.append({
                "sentence": example.sentence,
                "gold": example.gold_evaluation,
                "prediction": None,
                "score": 0.0
            })
    
    # Calculate and print overall results
    successful_examples = sum(1 for r in results if r["prediction"] is not None)
    total_score = sum(r["score"] for r in results if r["prediction"] is not None)
    average_score = total_score / len(results) if results else 0
    
    print("\n----- FINAL RESULTS -----")
    print(f"Successfully processed: {successful_examples}/{len(results)} examples")
    print(f"Average score (all examples): {average_score:.2f} ({average_score*100:.2f}%)")
    
    if successful_examples > 0:
        average_success_score = total_score / successful_examples
        print(f"Average score (successful examples only): {average_success_score:.2f} ({average_success_score*100:.2f}%)")
