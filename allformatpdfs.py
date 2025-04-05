import dspy
import json
import os
import traceback
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

def extract_text_with_pymupdf(file_path):
    """Try to extract text directly from PDF using PyMuPDF"""
    try:
        print(f"Attempting direct text extraction with PyMuPDF: {file_path}")
        document = fitz.open(file_path)
        
        # Check if document is encrypted
        if document.is_encrypted:
            print("Document is encrypted, attempting to decrypt with empty password")
            # Try with empty password
            if document.authenticate(""):
                print("Decryption successful")
            else:
                print("Document is password protected and could not be decrypted")
                return ""
        
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text()
            text += page_text
            
        document.close()
        
        # If we got minimal text, it might be a scanned PDF
        if len(text.strip()) < 100:
            print("Minimal text extracted, might be a scanned document")
            return ""
            
        print(f"Successfully extracted {len(text)} characters directly")
        return text
        
    except Exception as e:
        print(f"PyMuPDF error: {e}")
        return ""

def extract_text_with_ocr(file_path):
    """Extract text using OCR (for scanned PDFs)"""
    try:
        # Import required libraries
        from pdf2image import convert_from_path
        import pytesseract
        
        print(f"Attempting OCR extraction: {file_path}")
        
        # Convert PDF to images
        images = convert_from_path(file_path)
        print(f"Converted PDF to {len(images)} images")
        
        # Process all images with OCR and join with newlines
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        
        print(f"Extracted {len(text)} characters using OCR")
        return text
        
    except Exception as e:
        print(f"OCR error: {e}")
        traceback.print_exc()
        return ""

def extract_text_from_pdf(file_path):
    """Extract text from any PDF format using the most appropriate method"""
    # First, try direct extraction (faster and usually better for native PDFs)
    text = extract_text_with_pymupdf(file_path)
    
    # If direct extraction failed or returned minimal text, try OCR
    if not text or len(text.strip()) < 100:
        print("Direct extraction failed or returned minimal text, trying OCR...")
        text = extract_text_with_ocr(file_path)
    
    # If we still have no text, we've failed
    if not text:
        print("Failed to extract text using any method")
    
    return text

def main():
    # Get resume file name via console input
    print("Enter the resume PDF filename (e.g., resume1.pdf):")
    resume_file = input().strip()
    
    # Check if the file exists
    if not os.path.exists(resume_file):
        print(f"Error: The file '{resume_file}' does not exist. Please check the filename and try again.")
        return
    
    # Extract text using the combined approach
    try:
        resume = extract_text_from_pdf(resume_file)
    except ImportError:
        print("Required packages missing. Please install with:")
        print("pip install pymupdf pytesseract pdf2image")
        print("For Linux: sudo apt install poppler-utils tesseract-ocr")
        return
    
    if not resume:
        print("Could not extract text from the PDF. Please ensure all dependencies are installed correctly.")
        return
    
    # Save extracted text for debugging
    with open(f"{os.path.splitext(resume_file)[0]}_extracted.txt", "w", encoding="utf-8") as f:
        f.write(resume)
    print(f"Extracted text saved to {os.path.splitext(resume_file)[0]}_extracted.txt")
    
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
    output_filename = f"evaluation_result_{os.path.splitext(os.path.basename(resume_file))[0]}.json"
    with open(output_filename, "w") as f:
        f.write(output)
    print(f"\nEvaluation saved to {output_filename}")

if __name__ == "__main__":
    main()
