import dspy
from typing import List
from pydantic import BaseModel

# Import the Azure LLM configuration from your config file
from config import azure_llm

# Configure DSPy to use the Azure LLM
dspy.settings.configure(lm=azure_llm)

class SubSkill(BaseModel):
    name: str
    experience: int
    topics: List[str]

class Skill(BaseModel):
    objective: str
    skills: List[SubSkill]

class SkillsGenerationSignature(dspy.Signature):
    """Generate a comprehensive skills profile with varied experience levels for each skill"""
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    
    # Output Field using the Pydantic model directly
    skills_output: Skill = dspy.OutputField(
        desc="Skills in the correct format with 'objective' and 'skills' fields. Each skill should have a different experience level, with some skills having more experience than others. The experience levels should be realistic given the total years of experience provided."
    )

def main():
job_descriptions = [
        "Senior Software Developer role requiring cloud and DevOps expertise, with a focus on AWS, containerization, CI/CD pipelines, and infrastructure as code",
        "Data Scientist position focusing on machine learning and AI analytics, with emphasis on deep learning, NLP, and production model deployment",
        "Product Management role for strategic technology products, requiring market analysis, roadmap development, and stakeholder management"
    ]
    
    # Corresponding experience levels
    experience_levels = [5, 7, 10]
    resume_generator = dspy.Predict(SkillsGenerationSignature)
    
    # Using range(0, 3) to process three inputs
    for i in range(0, 3):
        job_desc = job_descriptions[i]
        exp = experience_levels[i]
        
        # Generate skills profile
        result = resume_generator(
            job_description=job_desc, 
            years_of_experience=exp
        )
        
        # Access and print the skills data
        skills_profile = result.skills_output
        print(f"\nProfile {i+1}:")
        print(skills_profile.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
