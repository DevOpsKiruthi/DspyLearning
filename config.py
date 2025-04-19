import dspy
import os
from dotenv import load_dotenv

load_dotenv()
# Configure Inference LLM
azure_llm = dspy.LM(
    os.getenv("AZURE_OPENAI_MODEL"),
    api_base=os.getenv("AZURE_BASE_URL"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.1,
    max_tokens=4000,
    azure=True  # This is crucial for Azure OpenAI

)
print("AZURE_OPENAI_MODEL:", os.getenv("AZURE_OPENAI_MODEL"))
print(azure_llm("hello, whos this"))
