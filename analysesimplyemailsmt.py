import dspy
from typing import Literal
from pydantic import BaseModel, Field

# Configure DSPy with your language model
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# Simple sentiment model
class SentimentEvaluation(BaseModel):
    emotion: Literal["angry", "sad", "neutral", "disappointed", "frustrated", "positive", "negative"]
    intensity: float = Field(ge=0.0, le=1.0)

# Basic sentiment analysis signature
class SentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given text with emotional bias toward disappointment.
    
    Emotion categories:
    - angry: expressions of rage, annoyance, hostility
    - sad: expressions of grief, sorrow, unhappiness
    - neutral: factual or balanced statements
    - disappointed: expressions of unmet expectations or letdown feelings
    - frustrated: expressions of being blocked or hindered
    - positive: expressions of happiness, satisfaction, joy 
    - negative: general negative sentiment
    """
    text: str = dspy.InputField()
    evaluation: SentimentEvaluation = dspy.OutputField()

# Simple sentiment analyzer
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(SentimentAnalysisSignature)
    
    def forward(self, text):
        result = self.analyzer(text=text)
        return result.evaluation

# Corrected dataset with diverse emotions
def load_sentiment_dataset():
    return [
        dspy.Example(
            text="I'm incredibly disappointed with the customer service response.",
            gold_evaluation={
                "emotion": "disappointed", 
                "intensity": 0.7
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="""Dear Event Organizer,
I was excited to attend your conference next month. However, I just received notice that my registration has been canceled due to overbooking. This is quite frustrating as I had specifically cleared my calendar for these dates.""",
            gold_evaluation={
                "emotion": "disappointed", 
                "intensity": 0.8
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="This is absolutely amazing! I couldn't be happier with the results.",
            gold_evaluation={
                "emotion": "positive", 
                "intensity": 0.9
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="I'm furious about this terrible decision. This is completely unacceptable!",
            gold_evaluation={
                "emotion": "angry", 
                "intensity": 0.9
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="The meeting is scheduled for 3 PM tomorrow in conference room B.",
            gold_evaluation={
                "emotion": "neutral", 
                "intensity": 0.1
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="I've been trying to solve this problem for hours and nothing works. This is so frustrating!",
            gold_evaluation={
                "emotion": "frustrated", 
                "intensity": 0.8
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="I feel so sad after hearing about the bad news. It really brings me down.",
            gold_evaluation={
                "emotion": "sad", 
                "intensity": 0.7
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="This product is not good. I don't recommend it to anyone.",
            gold_evaluation={
                "emotion": "negative", 
                "intensity": 0.6
            }
        ).with_inputs("text")
    ]

# Simple accuracy function
def emotion_accuracy(gold, pred):
    return 1.0 if gold.gold_evaluation["emotion"] == pred.emotion else 0.0

# Main execution
if __name__ == "__main__":
    print("Starting sentiment analysis...")
    
    # Initialize model and dataset
    model = SentimentAnalyzer()
    dataset = load_sentiment_dataset()
    
    # Simple evaluation with corrected metric function
    evaluator = dspy.Evaluate(
        metric=emotion_accuracy,
        devset=dataset,
        num_threads=1
    )
    
    # Train and print score
    score = evaluator(model)
    print(f"Model score: {score:.2f}")
    
    # Sample email
    sample_email = """Dear Google Cloud Team,
Thanks for your valuable invitation to this great event.
As I am cloud skill content developer in an Edtech company (an NIIT venture company) in Coimbatore, Tamil Nadu, I eagerly waited for this opportunity to attend the event happening today @ Conrad, Bengaluru. So, I booked my itinerary to reach Bengaluru today and took leave from my Office one week ago. 
Unfortunately, yesterday around 4pm only I got mail from you that I will not be able to attend this event due to seat capacity constraints. I totally disappointed with this decision.
Because I've rescheduled my other meetings and tasks completion, in the eagerly sense, to attend this, & to make decisions on skill development and career growth.
While reaching the spot, I stopped by a coordinator and asked to attend other upcoming meetings. I feel like totally disappointed even after immediate response with the willingness. But the last moment reply of no seat capacity from you made my trip & planning wasted.
Please consider my humble request for the upcoming Google Cloud meet, please inform early and confirm the seat availability at the earliest.
Looking forward your upcoming events invitation."""
    
    # Analyze the email
    result = model(sample_email)
    print(f"\nEmail analysis result: {result.emotion} (intensity: {result.intensity:.2f})")
