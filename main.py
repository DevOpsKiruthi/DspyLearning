import os
import dspy

# Directly setting values instead of using environment variables
llm = dspy.LM(
    model='gpt-4o',  
    #api_base='https://dspyaccount.openai.azure.com/', 
    api_base="https://rkiru-m8o4hmr4-swedencentral.cognitiveservices.azure.com/",
    api_key='9XNoNXY2xCr8NIrbkWCnbgmK5D8J89qfHIL3H8u8fAay4yyqyWH7JQQJ99BCAC4f1cMXJ3w3AAABACOGzJd1',  
    api_version='2021-04-30',  
    temperature=0.14,  # Example temperature value
    max_tokens=4096,  # Max tokens to generate
)

# Configure dspy with the LLM instance
dspy.configure(lm=llm)
print(llm("Hello"))
