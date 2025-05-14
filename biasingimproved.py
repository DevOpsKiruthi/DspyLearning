import dspy
from typing import List, Literal
from pydantic import BaseModel, Field

# Configure DSPy with your language model
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# Refined sentiment model with clear emotion categories
class SentimentEvaluation(BaseModel):
    emotion: Literal["angry", "sad", "neutral", "disappointed", "frustrated", "positive", "negative"]
    intensity: float = Field(ge=0.0, le=1.0)  # Range from 0 to 1

# Enhanced sentiment analysis signature with better guidelines
class SentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given sentence with nuanced understanding.
    
    Guidelines:
    1. Read the sentence carefully and identify the primary emotion expressed
    2. Consider context, tone, word choice, and emotional undertones
    3. Assess emotional intensity from 0.0 (minimal) to 1.0 (extremely intense)
    4. Choose only one emotion that best represents the overall sentiment
    5. Be consistent in your interpretation of similar patterns
    
    Emotion categories:
    - angry: expressions of rage, annoyance, hostility or indignation
    - sad: expressions of grief, sorrow, melancholy or unhappiness
    - neutral: factual, objective or balanced statements with minimal emotion
    - disappointed: expressions of unmet expectations or letdown feelings
    - frustrated: expressions of being blocked, hindered or unable to progress
    - positive: expressions of happiness, satisfaction, joy or optimism
    - negative: general negative sentiment not fitting other categories
    """
    sentence: str = dspy.InputField(desc="Sentence to analyze for sentiment")
    evaluation: SentimentEvaluation = dspy.OutputField(
        desc="Detailed sentiment evaluation with emotion category and intensity score"
    )

# Improved sentiment analyzer that uses more robust reasoning
class EnhancedSentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using Predict instead of ChainOfThought to potentially get more consistent results
        self.analyzer = dspy.ChainOfThought(SentimentAnalysisSignature)
    
    def forward(self, sentence):
        # Process the sentence and return the evaluation
        result = self.analyzer(sentence=sentence)
        return result.evaluation

# Create a more diverse and realistic dataset with edge cases
def load_sentiment_dataset():
    return [
        dspy.Example(
            sentence="The meeting was unexpectedly rescheduled without prior notice.",
            gold_evaluation={
                "emotion": "frustrated", 
                "intensity": 0.4
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="I'm incredibly disappointed with the customer service response.",
            gold_evaluation={
                "emotion": "disappointed", 
                "intensity": 0.7
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="The team successfully completed the project ahead of schedule.",
            gold_evaluation={
                "emotion": "positive", 
                "intensity": 0.6
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="After waiting for hours, no one addressed my concerns.",
            gold_evaluation={
                "emotion": "angry", 
                "intensity": 0.8
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="The policy seems fair and reasonable.",
            gold_evaluation={
                "emotion": "neutral", 
                "intensity": 0.1
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="I'm deeply frustrated by the lack of communication.",
            gold_evaluation={
                "emotion": "frustrated", 
                "intensity": 0.6
            }
        ).with_inputs("sentence"),
        
        # Additional examples for better coverage
        dspy.Example(
            sentence="This is just a simple factual statement about the weather.",
            gold_evaluation={
                "emotion": "neutral", 
                "intensity": 0.0
            }
        ).with_inputs("sentence"),
        
        dspy.Example(
            sentence="I am absolutely furious about how they handled this situation!",
            gold_evaluation={
                "emotion": "angry", 
                "intensity": 0.9
            }
        ).with_inputs("sentence")
    ]

# Improved metric with more precise matching logic
def sentiment_accuracy_metric(gold, pred):
    """
    Refined sentiment evaluation metric with better handling of emotion similarity
    and intensity scoring
    """
    if not hasattr(pred, 'emotion') or not hasattr(pred, 'intensity'):
        print("Prediction missing required attributes")
        return 0.0
    
    # Define emotion similarity matrix - which emotions are related
    emotion_similarity = {
        ("angry", "frustrated"): 0.7,
        ("frustrated", "angry"): 0.7,
        ("sad", "disappointed"): 0.7,
        ("disappointed", "sad"): 0.7,
        ("negative", "angry"): 0.5,
        ("negative", "sad"): 0.5,
        ("negative", "disappointed"): 0.6,
        ("negative", "frustrated"): 0.6,
        ("positive", "neutral"): 0.3,
        ("neutral", "positive"): 0.3
    }
    
    # Extract gold values
    gold_emotion = gold.gold_evaluation["emotion"].lower()
    gold_intensity = gold.gold_evaluation["intensity"]
    
    # Extract prediction values
    pred_emotion = pred.emotion.lower()
    pred_intensity = float(pred.intensity)
    
    # Calculate emotion score
    if gold_emotion == pred_emotion:
        emotion_score = 1.0
    else:
        # Check for similar emotions using the similarity matrix
        emotion_key = (gold_emotion, pred_emotion)
        emotion_score = emotion_similarity.get(emotion_key, 0.0)
    
    # Calculate intensity score - more forgiving for small differences
    intensity_diff = abs(gold_intensity - pred_intensity)
    intensity_score = max(0, 1.0 - (intensity_diff * 1.5))  # Penalize intensity differences
    
    # Weighted score (emotion is more important than exact intensity)
    final_score = (emotion_score * 0.7) + (intensity_score * 0.3)
    
    return final_score

# Optimize the model with DSPy techniques
def optimize_model(model, train_set):
    # Use Teleprompter to improve the model
    teleprompter = dspy.teleprompt.Teleprompter(
        task_model=model,
        metric=sentiment_accuracy_metric,
        trainset=train_set,
    )
    
    # Run bootstrap optimization
    optimized_model = teleprompter.bootstrap(
        num_bootstrapped_demos=3,
        max_bootstrapped_demos=5
    )
    
    return optimized_model

def train_and_evaluate_model():
    # Prepare dataset
    dataset = load_sentiment_dataset()
    
    # Split dataset - use more for training
    train_size = len(dataset) * 3 // 4
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    
    # Initialize model
    model = EnhancedSentimentAnalyzer()
    
    # Evaluate initial model
    initial_evaluator = dspy.Evaluate(
        metric=sentiment_accuracy_metric,
        devset=test_set,
        num_threads=1,
        display_progress=True
    )
    
    initial_score = initial_evaluator(model)
    print(f"Initial Model Performance: {initial_score*100:.2f}%")
    
    # Optional: Optimize the model (comment out if not needed)
    # optimized_model = optimize_model(model, train_set)
    # model = optimized_model  # Use the optimized model
    
    # Final evaluation
    final_evaluator = dspy.Evaluate(
        metric=sentiment_accuracy_metric,
        devset=test_set,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    final_score = final_evaluator(model)
    print(f"Final Model Performance: {final_score*100:.2f}%")
    
    return model

# Test function to examine individual predictions
def test_individual_examples(model):
    test_sentences = [
        "The policy seems fair and reasonable.",
        "I'm deeply frustrated by the lack of communication.",
        "This product exceeded all my expectations!",
        "The customer service representative was unhelpful and rude."
    ]
    
    print("\nTesting individual examples:")
    for sentence in test_sentences:
        result = model(sentence)
        print(f"Sentence: {sentence}")
        print(f"Prediction: emotion='{result.emotion}' intensity={result.intensity}")
        print("-" * 50)

# Main execution
if __name__ == "__main__":
    # Train and evaluate the model
    trained_model = train_and_evaluate_model()
    
    # Test on individual examples
    test_individual_examples(trained_model)
