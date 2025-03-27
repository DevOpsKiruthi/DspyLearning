import dspy
from typing import List, Dict, Literal
from config import azure_llm

# Configure the global language model
dspy.settings.configure(lm=azure_llm)

class ResumeGenerationSignature(dspy.Signature):
    """
    Generate a comprehensive resume based on job description and personal details
    """
    # Input Fields
    full_name: str = dspy.InputField(desc="Candidate's full name")
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    location: str = dspy.InputField(desc="Current location")
    
    # Output Fields
    objective: str = dspy.OutputField(desc="Professional objective statement")
    skills: str = dspy.OutputField(desc="JSON-formatted skills list with name, experience, and topics")
    work_experience: str = dspy.OutputField(desc="JSON-formatted work experience")
    education: str = dspy.OutputField(desc="JSON-formatted educational background")
    certifications: str = dspy.OutputField(desc="JSON-formatted professional certifications")

# Resume Builder Module with Chain of Thought
class ResumeBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.resume_generator = dspy.ChainOfThought(ResumeGenerationSignature)
    
    def forward(self, **kwargs):
        """
        Generate a comprehensive resume
        
        :param kwargs: Input parameters for resume generation
        :return: Generated resume details
        """
        # Generate resume using Chain of Thought
        resume = self.resume_generator(**kwargs)
        return resume

# Main Execution
def main():
    import json

    # Initialize Resume Builder
    resume_builder = ResumeBuilder()
    
    # Example Job Description
    job_description = """
    We are seeking a skilled Software Engineer with expertise in:
    - React and modern JavaScript frameworks
    - Cloud infrastructure (AWS/Azure)
    - Microservices architecture
    - DevOps practices
    
    Responsibilities:
    - Design and implement scalable web applications
    - Collaborate with cross-functional teams
    - Implement best practices in software development
    """
    
    # Generate Resume
    resume_result = resume_builder.forward(
        full_name="Jane Doe",
        job_description=job_description,
        years_of_experience=5,
        location="San Francisco, CA"
    )
    
    # Print Generated Resume Components
    print("📄 Generated Resume:\n")
    print("🎯 Objective:")
    print(resume_result.objective)
    
    print("\n💡 Skills:")
    try:
        skills = json.loads(resume_result.skills)
        for skill in skills:
            print(f"- {skill.get('name', 'N/A')} (Experience: {skill.get('experience', 'N/A')})")
            print(f"  Topics: {', '.join(skill.get('topics', []))}")
    except json.JSONDecodeError:
        print("Unable to parse skills JSON")
    
    print("\n💼 Work Experience:")
    try:
        work_exp = json.loads(resume_result.work_experience)
        for exp in work_exp:
            print(f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
    except json.JSONDecodeError:
        print("Unable to parse work experience JSON")

if __name__ == "__main__":
    main()