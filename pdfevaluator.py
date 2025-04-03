import dspy
import json
from typing import List
from pydantic import BaseModel
import fitz  # PyMuPDF

# Import the Azure LLM configuration from your config file
from config import azure_llm
# Configure DSPy to use the Azure LLM
dspy.settings.configure(lm=azure_llm)

class AIReview(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    fit_score: int
    comments: str

class CandidateEvaluation(BaseModel):
    name: str
    selected: bool
    not_selected: bool
    ai_review: AIReview

class ResumeEvaluationSignature(dspy.Signature):
    """Evaluate candidate resume against job description with additional expectations"""
    resume: str = dspy.InputField(desc="Candidate's full resume text")
    job_description: str = dspy.InputField(desc="Complete job description")
    what_we_are_expecting: str = dspy.InputField(desc="Additional comments or expectations for the role")
    
    # Output field using the Pydantic model
    evaluation: CandidateEvaluation = dspy.OutputField(
        desc="Evaluation including candidate's name extracted from resume, selection decision (either selected=True or not_selected=True), and detailed AI review with strengths, weaknesses, fit score, and comments"
    )

# Create the evaluator using dspy with ChainOfThought to expose reasoning
resume_evaluator = dspy.ChainOfThought(ResumeEvaluationSignature)

def read_pdf(file_path):
    """Read content from a PDF file using PyMuPDF."""
    try:
        document = fitz.open(file_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def main():
    # File path for resume
    resume_file = "resume.pdf"
    
    # Read resume from PDF file
    resume = read_pdf(resume_file)
    if not resume:
        print("Could not read the resume PDF. Please check the file path.")
        return
    
    # Get job description and expectations as console input
    print("Enter job description (press Enter twice when done):")
    job_description_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        job_description_lines.append(line)
    job_description = "\n".join(job_description_lines)
    
    print("\nEnter additional expectations (press Enter twice when done):")
    expectations_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        expectations_lines.append(line)
    expectations = "\n".join(expectations_lines)
    
    # Check if all inputs were provided
    if not all([resume, job_description, expectations]):
        print("One or more inputs are missing. Please provide all required information.")
        return
    
    print("\nProcessing evaluation...")
    
    # Generate evaluation with reasoning
    result = resume_evaluator(
        resume=resume,
        job_description=job_description,
        what_we_are_expecting=expectations
    )
    
    # Print the structured output
    evaluation = result.evaluation
    output = evaluation.model_dump_json(indent=2)
    print("\nEvaluation Result:")
    print(output)
    
    # Save the output to a JSON file
    with open("evaluation_result.json", "w") as f:
        f.write(output)
    print("\nEvaluation saved to evaluation_result.json")

if __name__ == "__main__":
    main()
