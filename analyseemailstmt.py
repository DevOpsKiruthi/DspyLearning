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
    reasoning: str = Field(default="")  # Added field to capture reasoning

# Enhanced sentiment analysis signature with better guidelines
class SentimentAnalysisSignature(dspy.Signature):
    """
    Analyze the sentiment of a given text (which might be a sentence, paragraph, or email) with nuanced understanding.
    
    Guidelines:
    1. Read the text carefully and identify the primary emotion expressed
    2. Consider context, tone, word choice, and emotional undertones
    3. Assess emotional intensity from 0.0 (minimal) to 1.0 (extremely intense)
    4. Choose only one emotion that best represents the overall sentiment
    5. Pay special attention to expressions of disappointment, frustration, or negative feelings
    6. For longer texts like emails, focus on the overall tone and main message
    7. Provide brief reasoning for your emotion classification
    
    Emotion categories:
    - angry: expressions of rage, annoyance, hostility or indignation
    - sad: expressions of grief, sorrow, melancholy or unhappiness
    - neutral: factual, objective or balanced statements with minimal emotion
    - disappointed: expressions of unmet expectations or letdown feelings
    - frustrated: expressions of being blocked, hindered or unable to progress
    - positive: expressions of happiness, satisfaction, joy or optimism
    - negative: general negative sentiment not fitting other categories
    """
    text: str = dspy.InputField(desc="Text to analyze for sentiment (can be a sentence, paragraph, or email)")
    evaluation: SentimentEvaluation = dspy.OutputField(
        desc="Detailed sentiment evaluation with emotion category, intensity score, and reasoning"
    )

# Improved sentiment analyzer using standard DSPy modules
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning on complex text
        self.analyzer = dspy.ChainOfThought(SentimentAnalysisSignature)
    
    def forward(self, text):
        # Process the text and return the evaluation
        result = self.analyzer(text=text)
        return result.evaluation

# Create a minimal dataset with just 2 examples for learning
def load_sentiment_dataset():
    return [
        dspy.Example(
            text="I'm incredibly disappointed with the customer service response.",
            gold_evaluation={
                "emotion": "disappointed", 
                "intensity": 0.7,
                "reasoning": "Direct expression of disappointment with strong intensity marker 'incredibly'"
            }
        ).with_inputs("text"),
        
        dspy.Example(
            text="""Dear Event Organizer,
I was excited to attend your conference next month. However, I just received notice that my registration has been canceled due to overbooking. I had already arranged travel and accommodations for this event. This is quite frustrating as I had specifically cleared my calendar for these dates.
Regards,
John""",
            gold_evaluation={
                "emotion": "disappointed", 
                "intensity": 0.8,
                "reasoning": "The email expresses clear disappointment about canceled registration after making specific arrangements"
            }
        ).with_inputs("text")
    ]

# Simple metric for sentiment evaluation with improved accuracy
def sentiment_accuracy_metric(gold, pred):
    """
    Enhanced sentiment evaluation metric with better scoring logic
    """
    if pred is None or not hasattr(pred, 'emotion') or not hasattr(pred, 'intensity'):
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
    
    # Extract gold and prediction values
    gold_emotion = gold.gold_evaluation["emotion"].lower()
    gold_intensity = gold.gold_evaluation["intensity"]
    pred_emotion = pred.emotion.lower()
    
    try:
        pred_intensity = float(pred.intensity)
    except (TypeError, ValueError):
        print(f"Warning: Invalid intensity value: {pred.intensity}")
        pred_intensity = 0.5  # Default fallback
    
    # Calculate emotion score with similarity considerations
    if gold_emotion == pred_emotion:
        emotion_score = 1.0
    else:
        # Check for similar emotions using the similarity matrix
        emotion_key = (gold_emotion, pred_emotion)
        emotion_score = emotion_similarity.get(emotion_key, 0.0)
    
    # Use a more forgiving intensity scoring
    intensity_diff = abs(gold_intensity - pred_intensity)
    intensity_score = max(0, 1.0 - (intensity_diff * 0.8))
    
    # Weighted score (emotion is more important than exact intensity)
    final_score = (emotion_score * 0.7) + (intensity_score * 0.3)
    
    return final_score

def train_and_evaluate_model():
    # Prepare dataset
    dataset = load_sentiment_dataset()
    
    # Initialize model
    model = SentimentAnalyzer()
    
    # Standard DSPy evaluation approach
    evaluator = dspy.Evaluate(
        metric=sentiment_accuracy_metric,
        devset=dataset,
        display_progress=True,
        display_table=True,
        num_threads=1  # Single-threaded to avoid rate limits
    )
    
    score = evaluator(model)
    print(f"Model Performance: {score*100:.2f}%")
    
    return model

# Analyze a specific email
def analyze_email(model, email_text):
    print("\nAnalyzing Email Sentiment:")
    print("-" * 60)
    print(f"Email Content (preview): {email_text[:100]}...")
    print("-" * 60)
    
    try:
        result = model(email_text)
        print(f"Primary Emotion: {result.emotion}")
        print(f"Intensity: {result.intensity:.2f} (on a scale of 0-1)")
        if hasattr(result, 'reasoning') and result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        
        print("\nSentiment Analysis Summary:")
        if result.intensity >= 0.8:
            intensity_level = "very strong"
        elif result.intensity >= 0.6:
            intensity_level = "strong"
        elif result.intensity >= 0.4:
            intensity_level = "moderate"
        else:
            intensity_level = "mild"
            
        print(f"This email expresses {intensity_level} {result.emotion} sentiment.")
        
        # Additional advice based on sentiment
        if result.emotion in ["disappointed", "frustrated", "angry", "negative"]:
            if result.intensity >= 0.7:
                print("\nRecommendation: This message indicates significant customer dissatisfaction.")
                print("Suggested Action: Consider prioritizing a thoughtful and timely response.")
        
        return result
    except Exception as e:
        print(f"Error analyzing email: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Starting email sentiment analysis...")
    
    # Sample email from the problem description
    sample_email = """Dear Google Cloud Team,
Thanks for your valuable invitation to this great event.
As I am cloud skill content developer in an Edtech company (an NIIT venture company) in Coimbatore, Tamil Nadu, I eagerly waited for this opportunity to attend the event happening today @ Conrad, Bengaluru. So, I booked my itinerary to reach Bengaluru today and took leave from my Office one week ago. 
Unfortunately, yesterday around 4pm only I got mail from you that I will not be able to attend this event due to seat capacity constraints. I totally disappointed with this decision.
Because I've rescheduled my other meetings and tasks completion, in the eagerly sense, to attend this, & to make decisions on skill development and career growth.
While reaching the spot, I stopped by a coordinator and asked to attend other upcoming meetings. I feel like totally disappointed even after immediate response with the willingness. But the last moment reply of no seat capacity from you made my trip & planning wasted.
Please consider my humble request for the upcoming Google Cloud meet, please inform early and confirm the seat availability at the earliest.
Looking forward your upcoming events invitation."""
    
    try:
        # Train and evaluate the model
        trained_model = train_and_evaluate_model()
        
        # Analyze the sample email
        result = analyze_email(trained_model, sample_email)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Try adjusting the parameters for better results.")
