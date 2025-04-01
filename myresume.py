import dspy
import json
from typing import List, Dict
from config import azure_llm

# Configure the global language model
dspy.settings.configure(lm=azure_llm)

class ResumeGenerationSignature(dspy.Signature):
    """
    Generate a comprehensive resume with detailed sections
    """
    # Input Fields
    full_name: str = dspy.InputField(desc="Candidate's full name")
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    location: str = dspy.InputField(desc="Current location")
    
    # Expanded Output Fields
    professional_summary: str = dspy.OutputField(desc="Detailed professional summary")
    professional_achievements: List[str] = dspy.OutputField(desc="List of professional achievements")
    skills: str = dspy.OutputField(desc="JSON-formatted skills with categories")
    work_history: str = dspy.OutputField(desc="JSON-formatted detailed work history")
    education: str = dspy.OutputField(desc="JSON-formatted educational background")
    certifications: List[str] = dspy.OutputField(desc="List of professional certifications")
    technical_proficiencies: str = dspy.OutputField(desc="JSON-formatted technical proficiencies")

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
    # Initialize Resume Builder
    resume_builder = ResumeBuilder()
    
    # Example Job Description
    job_description = """
    We are seeking an experienced Cloud and Training Professional with:
    - Expertise in Azure, AWS, and GCP Cloud Services
    - Strong background in technical content creation
    - Experience in training and educational technology
    - Proficiency in software development and cloud technologies
    
    Responsibilities:
    - Develop technical training content
    - Provide cloud computing training
    - Create educational resources and assessments
    """
    
    # Generate Resume
    resume_result = resume_builder.forward(
        full_name="Jane Doe",
        job_description=job_description,
        years_of_experience=5,
        location="London, UK"
    )
    
    # Print Generated Resume Components
    print("📄 Generated Resume:\n")
    
    # Professional Summary
    print("🌟 Professional Summary:")
    print(resume_result.professional_summary)
    
    # Professional Achievements
    print("\n🏆 Professional Achievements:")
    for achievement in resume_result.professional_achievements:
        print(f"ü  {achievement}")
    
    # Skills
    print("\n💡 Skills:")
    try:
        skills = json.loads(resume_result.skills)
        for category, skill_list in skills.items():
            print(f"·  **{category}:** {', '.join(skill_list)}")
    except json.JSONDecodeError:
        print("Unable to parse skills JSON")
    
    # Technical Proficiencies
    print("\n🖥️ Technical Proficiencies:")
    try:
        tech_skills = json.loads(resume_result.technical_proficiencies)
        for category, proficiencies in tech_skills.items():
            print(f"·  **{category}:** {', '.join(proficiencies)}")
    except json.JSONDecodeError:
        print("Unable to parse technical proficiencies JSON")
    
    # Work History
    print("\n💼 Work History:")
    try:
        work_history = json.loads(resume_result.work_history)
        for job in work_history:
            print(f"\n{job.get('job_title', 'N/A')} at {job.get('company', 'N/A')}")
            print(f"Period: {job.get('period', 'N/A')}")
            for responsibility in job.get('responsibilities', []):
                print(f"¨  {responsibility}")
    except json.JSONDecodeError:
        print("Unable to parse work history JSON")
    
    # Education
    print("\n🎓 Education:")
    try:
        education = json.loads(resume_result.education)
        for degree in education:
            print(f"{degree.get('degree', 'N/A')}, {degree.get('institution', 'N/A')}")
    except json.JSONDecodeError:
        print("Unable to parse education JSON")
    
    # Certifications
    print("\n📜 Certifications:")
    for cert in resume_result.certifications:
        print(f"·  {cert}")

if __name__ == "__main__":
    main()
