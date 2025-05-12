import dspy
from typing import List, Literal
from pydantic import BaseModel, Field
import json

# Configure DSPy with your language model
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# More flexible sentiment model
class SentimentEvaluation(BaseModel):
    emotion: Literal["angry", "sad", "neutral", "disappointed", "frustrated", "positive", "negative"]
    intensity: float = Field(ge=0.0, le=1.0)  # Full range from 0 to 1

# More generalized sentiment analysis signature
class ImprovedSentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given sentence with nuanced understanding.
    
    Guidelines:
    1. Consider context, tone, and emotional undertones
    2. Assess emotional intensity from 0.0 (neutral) to 1.0 (extremely intense)
    3. Choose the most appropriate emotion that captures the sentiment
    4. Be consistent but not overly rigid
    
    Emotion categories:
    - angry: frustration, rage, annoyance
    - sad: grief, disappointment, melancholy
    - neutral: factual, balanced statements
    - disappointed: unmet expectations, mild frustration
    - frustrated: blocked progress, mild anger
    - positive: happiness, satisfaction, joy
    - negative: general negative sentiment
    """
    sentence: str = dspy.InputField(desc="Sentence to analyze")
    evaluation: SentimentEvaluation = dspy.OutputField(
        desc="Sentiment evaluation with emotion and intensity"
    )

# More robust sentiment analysis module
class FlexibleSentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ImprovedSentimentAnalysisSignature)
    
    def forward(self, sentence):
        result = self.analyze(sentence=sentence)
        return result.evaluation

# Create a more diverse and realistic dataset
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
        ).with_inputs("sentence")
    ]

# Improved metric that allows more nuanced matching
def nuanced_sentiment_metric(gold, pred):
    """
    More flexible sentiment evaluation metric
    """
    if not hasattr(pred, 'emotion') or not hasattr(pred, 'intensity'):
        return 0.0
    
    # Emotion similarity mapping
    emotion_similarity = {
        ("angry", "frustrated"): 0.8,
        ("frustrated", "angry"): 0.8,
        ("sad", "disappointed"): 0.7,
        ("disappointed", "sad"): 0.7,
        ("negative", "angry"): 0.6,
        ("negative", "sad"): 0.6
    }
    
    # Exact emotion match
    if gold.gold_evaluation["emotion"].lower() == pred.emotion.lower():
        emotion_score = 1.0
    else:
        # Check for similar emotions
        emotion_key = (gold.gold_evaluation["emotion"].lower(), pred.emotion.lower())
        emotion_score = emotion_similarity.get(emotion_key, 0.0)
    
    # Intensity difference scoring
    gold_intensity = gold.gold_evaluation["intensity"]
    intensity_diff = abs(gold_intensity - pred.intensity)
    intensity_score = max(0, 1.0 - intensity_diff)
    
    # Weighted score
    return (emotion_score * 0.6) + (intensity_score * 0.4)

def train_and_evaluate_model():
    # Prepare dataset
    dataset = load_sentiment_dataset()
    
    # Split dataset
    train_set = dataset[:4]
    test_set = dataset[4:]
    
    # Initialize model
    model = FlexibleSentimentAnalyzer()
    
    # Evaluate initial model
    initial_evaluator = dspy.Evaluate(
        metric=nuanced_sentiment_metric,
        devset=test_set,
        num_threads=1,
        display_progress=True
    )
    
    initial_score = initial_evaluator(model)
    print(f"Initial Model Performance: {initial_score*100:.2f}%")
    
    # Optional: Add optimization steps if needed
    
    # Final evaluation
    final_evaluator = dspy.Evaluate(
        metric=nuanced_sentiment_metric,
        devset=test_set,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    final_score = final_evaluator(model)
    print(f"Final Model Performance: {final_score*100:.2f}%")
    
    return model

# Main execution
if __name__ == "__main__":
    trained_model = train_and_evaluate_model()
