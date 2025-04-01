import dspy
import json
import time
from typing import Dict, Any

# Import the Azure LLM configuration from your config file
from config import azure_llm

# Configure DSPy to use the Azure LLM
dspy.settings.configure(lm=azure_llm)

class SkillsGenerationSignature(dspy.Signature):
    """Generate a comprehensive and structured skills profile based on job description and experience"""
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    
    skills: str = dspy.OutputField(
        desc="JSON-formatted array of skills with name, experience, and topics. " 
             "Each skill should include: name (string), experience (string), topics (array of strings). " 
             "Ensure skills are directly relevant to the job description."
    )

class SkillsGenerator(dspy.Module):
    def __init__(self, max_retries=3, retry_delay=30):
        super().__init__()
        self.skill_generator = dspy.ChainOfThought(SkillsGenerationSignature)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _safe_generate_skills(self, method, job_description, years_of_experience):
        """
        Safe method to generate skills with retry logic for rate limits
        """
        for attempt in range(self.max_retries):
            try:
                result = method(
                    job_description=job_description, 
                    years_of_experience=years_of_experience
                )
                
                # Attempt to parse skills JSON
                skills = json.loads(result.skills)
                return skills
            
            except json.JSONDecodeError:
                print(f"Attempt {attempt + 1}: JSON parsing failed")
                if attempt == self.max_retries - 1:
                    return {"error": "Could not parse skills JSON"}
            
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    print(f"Rate limit hit. Waiting {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Unexpected error: {e}")
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def predict(self, job_description: str, years_of_experience: int) -> Dict[str, Any]:
        """
        Predict skills with additional error handling
        """
        return self._safe_generate_skills(
            self.skill_generator, 
            job_description, 
            years_of_experience
        )
    
    def chain_of_thoughts(self, job_description: str, years_of_experience: int) -> Dict[str, Any]:
        """
        More detailed reasoning process for skills generation
        """
        reasoning_prompt = f"""
        Carefully analyze the following job description considering {years_of_experience} years of professional experience:
        
        Job Description: {job_description}
        
        Reasoning Steps:
        1. Identify core skill domains required by the job
        2. Match candidate's experience level to skill depth and complexity
        3. Generate specific, measurable topics within each skill category
        4. Ensure skills demonstrate direct relevance to job requirements
        5. Prioritize skills that showcase professional growth and expertise
        
        Provide the output as a JSON-formatted array of skills.
        """
        
        return self._safe_generate_skills(
            lambda job_description, years_of_experience: 
                self.skill_generator(job_description=reasoning_prompt, years_of_experience=years_of_experience),
            job_description, 
            years_of_experience
        )

def print_skills(skills_data: Dict[str, Any]):
    """
    Utility function to print skills in a readable format
    """
    if 'error' in skills_data:
        print(f"Error: {skills_data['error']}")
        return
    
    print("\nSKILLS:")
    for skill in skills_data:
        print(f"- {skill.get('name', 'Unnamed Skill')}")
        print(f"  Experience: {skill.get('experience', 'Not specified')}")
        print(f"  Topics: {', '.join(skill.get('topics', []))}\n")

def main():
    job_descriptions = [
        "Senior Software Developer role requiring cloud and DevOps expertise",
        "Data Scientist position focusing on machine learning and AI analytics",
        "Product Management role for strategic technology products"
    ]
    
    experience_levels = [3, 5, 7]
    
    skills_generator = SkillsGenerator(max_retries=5, retry_delay=35)
    
    for job_desc, exp in zip(job_descriptions, experience_levels):
        print(f"\nJob Description: {job_desc}")
        print(f"Total Experience: {exp} years")
        
        print("\n--- Predict Method ---")
        predict_skills = skills_generator.predict(job_desc, exp)
        print_skills(predict_skills)
        
        print("\n--- Chain of Thoughts Method ---")
        cot_skills = skills_generator.chain_of_thoughts(job_desc, exp)
        print_skills(cot_skills)
        print("-" * 50)

if __name__ == "__main__":
    main()