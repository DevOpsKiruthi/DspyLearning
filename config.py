import dspy
# Configure Inference LLM
azure_llm = dspy.LM(
    "azure/gpt-4o",
    api_base="https://rkiru-m8o4hmr4-swedencentral.openai.azure.com/",
    api_version="2024-05-01-preview",
    api_key="BuNHnTKkDXHtUCye4Q4TrRsxNiX45j48NzmtDdwLOClBcAhvOBIkJQQJ99BCACfhMk5XJ3w3AAAAACOGbPCh",
    temperature=0.1,
    max_tokens=4000,
)

print(azure_llm("hello, whos this"))
