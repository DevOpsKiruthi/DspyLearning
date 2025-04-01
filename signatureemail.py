(myenv) kiruthika@DspyMachine:~/mydspy$ python signatureemail.py
["Hello! I'm ChatGPT, your friendly AI assistant. How can I help you today? 😊"]

📌 Please enter the email details:

From: test
To: test
Subject: testing
Start Date: 20-02-2025
End Date: 22-02-2026
Keywords (comma-separated): test1,test2
Tone (formal, semi-formal, casual, friendly): formal
Would you like to include a custom signature? (yes/no): no

📧 **Generated Email:**

### Thought Process:

1. **Purpose of the Email**: The subject line "testing" and the keywords "test1, test2" suggest that this email is likely a placeholder or a test email. However, the inclusion of a start date (20-02-2025) and an end date (22-02-2026) implies that the email might be referencing a timeline or a project/event spanning this period.

2. **Structure of the Email**:
   - **Opening**: Acknowledge the recipient and provide context for the email.
   - **Body**: Clearly state the purpose of the email, referencing the provided dates and keywords. If this is a test email, explicitly mention that it is for testing purposes.
   - **Closing**: End with a polite closing, offering to clarify or provide additional information if needed.

3. **Tone**: Since this is a formal email, the tone should be professional, concise, and clear.

4. **Key Details to Include**:
   - Mention the start and end dates.
   - Reference the keywords "test1" and "test2" to ensure they are acknowledged.
   - Clarify the purpose of the email (e.g., testing or placeholder).

---

### Final Email Body:

**Subject**: Testing

Dear [Recipient's Name],

I hope this message finds you well.

This email is being sent as part of a testing process to ensure proper communication and functionality. The timeline for this test spans from **20 February 2025** to **22 February 2026**. During this period, the focus will be on the following key areas: **test1** and **test2**.

Please let me know if you require any additional details or if there are specific aspects you would like me to address during this testing phase.

Thank you for your time and attention.

Best regards,
[Your Full Name]
[Your Position]
[Your Contact Information]

---

### Reasoning for the Final Email:
- The opening provides a polite and professional introduction.
- The body clearly states the purpose of the email (testing) and references the provided dates and keywords.
- The closing invites further communication, ensuring the recipient feels included and informed.

Best regards,
testimport dspy
from config import azure_llm  # Import Azure LLM from config.py

# DsPy module for email generation with tone
class EmailWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lm = azure_llm

    def forward(self, from_name, to_name, subject, start_date, end_date, keywords, tone, custom_signature=None):
        # Step 1: Generate the thought process for the email body
        prompt = (
            f"Think step-by-step and generate a {tone} email using the following details:\n"
            f"- From: {from_name}\n"
            f"- To: {to_name}\n"
            f"- Subject: {subject}\n"
            f"- Start Date: {start_date}\n"
            f"- End Date: {end_date}\n"
            f"- Keywords: {keywords}\n\n"
            f"Write a clear, concise, and professional email body message.\n"
            f"Provide reasoning for how the message should be structured based on the given details.\n"
            f"Then, generate a final email body based on the thought process."
        )

        # Predict email body with chain of thoughts
        email_body_thoughts = self.lm(prompt)

        # Step 2: Generate signature based on reasoning
        if custom_signature:
            signature_prompt = f"Based on the user's input, generate a professional email signature for {from_name}. Think through the best format and tone."
            signature = self.lm(signature_prompt)
            email_body = email_body_thoughts + f"\n\n{signature}"
        else:
            # Predict default signature if no custom signature is provided
            default_signature = f"\n\nBest regards,\n{from_name}"
            # Convert the list to a string
            email_body = ''.join(email_body_thoughts) + default_signature


        return email_body


# ✅ Prompting for user inputs
print("\n📌 Please enter the email details:\n")
from_name = input("From: ")
to_name = input("To: ")
subject = input("Subject: ")
start_date = input("Start Date: ")
end_date = input("End Date: ")
keywords = input("Keywords (comma-separated): ")
tone = input("Tone (formal, semi-formal, casual, friendly): ")
custom_signature = input("Would you like to include a custom signature? (yes/no): ").strip().lower() == 'yes'

# Generate the email
email_writer = EmailWriter()
response = email_writer.forward(from_name, to_name, subject, start_date, end_date, keywords, tone, custom_signature)

# Display the generated email
print("\n📧 **Generated Email:**\n")
print(response)
