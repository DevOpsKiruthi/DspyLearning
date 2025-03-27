import dspy
import json
from config import azure_llm

# Configure the global language model
dspy.settings.configure(lm=azure_llm)

class ResumeGenerationSignature(dspy.Signature):
    """Generate a comprehensive resume with detailed sections"""
    full_name: str = dspy.InputField(desc="Candidate's full name")
    job_description: str = dspy.InputField(desc="Detailed job description")
    years_of_experience: int = dspy.InputField(desc="Total years of professional experience")
    location: str = dspy.InputField(desc="Current location")
    
    # Output Fields
    professional_summary: str = dspy.OutputField(desc="Detailed professional summary")
    skills: str = dspy.OutputField(desc="JSON-formatted skills with categories")
    work_history: str = dspy.OutputField(desc="JSON-formatted detailed work history")
    education: str = dspy.OutputField(desc="JSON-formatted educational background")
    certifications: str = dspy.OutputField(desc="JSON-formatted professional certifications")

class ResumeBuilder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.resume_generator = dspy.ChainOfThought(ResumeGenerationSignature)
    
    def forward(self, **kwargs):
        return self.resume_generator(**kwargs)

def print_section(title, content, parse_json=False):
    """
    Utility function to print resume sections with consistent formatting
    
    :param title: Section title
    :param content: Content to print
    :param parse_json: Whether to parse content as JSON
    """
    print(f"\n{title.upper()}:")
    
    if parse_json:
        try:
            parsed_content = json.loads(content)
            if isinstance(parsed_content, dict):
                for category, items in parsed_content.items():
                    print(f"• {category}: {', '.join(items)}")
            elif isinstance(parsed_content, list):
                for item in parsed_content:
                    print(f"• {item}")
        except (json.JSONDecodeError, TypeError):
            print(f"Could not parse {title.lower()} content")
    else:
        print(content)

def main():
    # Initialize Resume Builder
    resume_builder = ResumeBuilder()
    
    # Example Job Description
    job_description = """
    Seeking a Cloud and Training Professional with:
    - Expertise in Azure, AWS, and GCP Cloud Services
    - Strong technical content creation skills
    - Experience in training and educational technology
    """
    
    # Generate Resume
    resume = resume_builder.forward(
        full_name="Jane Doe",
        job_description=job_description,
        years_of_experience=5,
        location="London, UK"
    )
    
    # Print Resume Sections
    print_section("Professional Summary", resume.professional_summary)
    print_section("Skills", resume.skills, parse_json=True)
    print_section("Work History", resume.work_history, parse_json=True)
    print_section("Education", resume.education, parse_json=True)
    print_section("Certifications", resume.certifications, parse_json=True)

if __name__ == "__main__":
    main()
