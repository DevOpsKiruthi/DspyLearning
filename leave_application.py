import dspy
from config import azure_llm  # Import Azure LLM from config.py

# DsPy module for email generation with tone
class EmailWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = azure_llm

    def forward(self, from_name, to_name, subject, start_date, end_date, keywords, tone):
        prompt = (
            f"Generate a {tone} email using the following details:\n"
            f"- From: {from_name}\n"
            f"- To: {to_name}\n"
            f"- Subject: {subject}\n"
	    f"- Start_Date: {start_date}\n"
	    f"- End_Date: {end_date}\n"
            f"- Keywords: {keywords}\n\n"
            f"Write a clear, concise, and professional email body message."
        )
        return self.lm(prompt)

# ✅ Prompting for user inputs
print("\n📌 Please enter the email details:\n")

from_name = input("From: ")
to_name = input("To: ")
subject = input("Subject: ")
start_date = input("Start_Date: ")
end_date = input("End_Date: ")
keywords = input("Keywords (comma-separated): ")
tone = input("Tone (formal, semi-formal, casual, friendly): ")

# Generate the email
email_writer = EmailWriter()
response = email_writer.forward(from_name, to_name, subject, start_date, end_date, keywords, tone)

# Display the generated email
print("\n📧 **Generated Email:**\n")
print(response)
