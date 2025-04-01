import dspy
from config import azure_llm  # Import the Azure LLM from your config file

# Configure the global language model
dspy.settings.configure(lm=azure_llm)

class SentimentClassification(dspy.Signature):
    """Classify the sentiment of a sentence"""
    sentence: str = dspy.InputField()
    sentiment: bool = dspy.OutputField()

# Create the classifier
classify = dspy.Predict(SentimentClassification)

# Example usage
sentence = "it's a charming and often affecting journey."
result = classify(sentence=sentence)

# Print the sentiment
print(result.sentiment)
