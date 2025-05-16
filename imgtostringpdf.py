import dspy
import json
import os
import traceback
from typing import List
from pydantic import BaseModel

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

def extract_text_from_pdf(file_path):
    """Extract text from PDF using OCR"""
    try:
        # Import required libraries
        from pdf2image import convert_from_path
        import pytesseract
        
        print(f"Processing PDF with OCR: {file_path}")
        
        # Convert PDF to images
        images = convert_from_path(file_path)
        print(f"Converted PDF to {len(images)} images")
        
        # Extract text from all images and join with newlines
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        
        print(f"Extracted {len(text)} characters using OCR")
        return text
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        traceback.print_exc()
        return ""

def main():
    # Get resume file name via console input
    print("Enter the resume PDF filename (e.g., resume1.pdf):")
    resume_file = input().strip()
    
    # Check if the file exists
    if not os.path.exists(resume_file):
        print(f"Error: The file '{resume_file}' does not exist.")
        return
    
    # Extract text from PDF using OCR
    resume = extract_text_from_pdf(resume_file)
    
    if not resume:
        print("Failed to extract text from the PDF.")
        return
    
    # Save extracted text for debugging
    txt_file = f"{os.path.splitext(resume_file)[0]}_extracted.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(resume)
    print(f"Extracted text saved to {txt_file}")
    
    # Get job description from user
    print("\nEnter job description (press Enter twice when done):")
    job_description_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        job_description_lines.append(line)
    job_description = "\n".join(job_description_lines)
    
    # Get additional expectations from user
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
        print("One or more inputs are missing.")
        return
    
    print("\nProcessing evaluation...")
    
    # Generate evaluation with reasoning
    try:
        result = resume_evaluator(
            resume=resume,
            job_description=job_description,
            what_we_are_expecting=expectations
        )
        
        # Print and save the structured output
        evaluation = result.evaluation
        output = evaluation.model_dump_json(indent=2)
        print("\nEvaluation Result:")
        print(output)
        
        # Save to JSON file
        output_filename = f"evaluation_{os.path.splitext(os.path.basename(resume_file))[0]}.json"
        with open(output_filename, "w") as f:
            f.write(output)
        print(f"\nEvaluation saved to {output_filename}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
