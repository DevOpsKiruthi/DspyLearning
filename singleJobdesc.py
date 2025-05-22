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
    """Generate a comprehensive skills profile as a JSON string"""
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    
    # Output Field using the Pydantic model directly
    skills_output: Skill = dspy.OutputField(
        desc="Skills in the correct format with 'objective' and 'skills' fields"
    )

def main():
    # Single job description and experience level
    job_description = "Senior Software Developer role requiring cloud and DevOps expertise"
    years_of_experience = 5
    
    # Create the predictor using dspy.Predict
    resume_generator = dspy.Predict(SkillsGenerationSignature)
    
    # Generate skills profile
    result = resume_generator(job_description=job_description, years_of_experience=years_of_experience)
    
    # Access and print the skills data
    skills_profile = result.skills_output
    print(skills_profile.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
