import dspy
import json
from typing import Dict, Any, List

# Import the Azure LLM configuration from your config file
from config import azure_llm

# Configure DSPy to use the Azure LLM
dspy.settings.configure(lm=azure_llm)

class SkillsGenerationSignature(dspy.Signature):
    """Generate a comprehensive skills profile as a JSON string"""
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    
    # Output Field specifying the exact JSON structure
    skills_output: str = dspy.OutputField(
        desc="Strict JSON string with this exact structure: "
             '{"objective": "Career objective as a clear, concise string", '
             '"skills": [{'
             '   "name": "Skill Name", '
             '   "experience": "X years", '
             '   "topics": ["Topic 1", "Topic 2"]'
             '}]}'
    )

class SkillsGenerator(dspy.Module):
    def __init__(self, max_retries=3):
        super().__init__()
        self.skill_generator = dspy.ChainOfThought(SkillsGenerationSignature)
        self.max_retries = max_retries
    
    def predict(self, job_description: str, years_of_experience: int) -> Dict[str, Any]:
        """
        Generate skills profile with strict JSON output
        """
        for _ in range(self.max_retries):
            try:
                # Generate skills using the signature
                result = self.skill_generator(
                    job_description=job_description, 
                    years_of_experience=years_of_experience
                )
                
                # Ensure the result is a valid JSON string
                skills_data = json.loads(result.skills_output)
                return skills_data
            
            except Exception:
                # Silent failure with fallback
                pass
        
        # Fallback output if generation fails
        return {
            "objective": "Professional skills development",
            "skills": []
        }

def main():
    # Example job descriptions with varying complexity
    job_descriptions = [
        "Senior Software Developer role requiring cloud and DevOps expertise",
        "Data Scientist position focusing on machine learning and AI analytics",
        "Product Management role for strategic technology products"
    ]
    
    # Corresponding experience levels
    experience_levels = [3, 5, 7]
    
    # Initialize Skills Generator
    skills_generator = SkillsGenerator()
    
    for job_desc, exp in zip(job_descriptions, experience_levels):
        # Generate skills profile
        skills_profile = skills_generator.predict(job_desc, exp)
        
        # Print the raw JSON output
        print(json.dumps(skills_profile, indent=2))

if __name__ == "__main__":
    main()