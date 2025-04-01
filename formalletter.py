import dspy
from typing import Literal
from config import azure_llm  # Import the Azure LLM from your config file

# Configure the global language model
dspy.settings.configure(lm=azure_llm)

class LeaveEmailSignature(dspy.Signature):
    """Generate a professional leave application email"""
    from_name: str = dspy.InputField(desc="Name of the sender")
    to_name: str = dspy.InputField(desc="Name of the recipient")
    start_date: str = dspy.InputField(desc="First day of leave")
    end_date: str = dspy.InputField(desc="Last day of leave")
    keywords: str = dspy.InputField(desc="Additional context or reasons for leave")
    tone: Literal['formal', 'semi-formal', 'casual'] = dspy.InputField(desc="Email tone")
    
    subject: str = dspy.OutputField(desc="Email subject line")
    body: str = dspy.OutputField(desc="Email body content")

# Create the email generator
generate_leave_email = dspy.Predict(LeaveEmailSignature)

# Example usage
result = generate_leave_email(
    from_name="John Doe",
    to_name="HR Manager",
    start_date="2024-07-15",
    end_date="2024-07-30",
    keywords="Family vacation, pre-planned trip",
    tone="formal"
)

# Print the generated email details
print("Subject:", result.subject)
print("\nBody:", result.body)
