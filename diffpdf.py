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

def read_pdf(file_path):
    """Read content from a PDF file using PyMuPDF with better error handling."""
    try:
        print(f"Attempting to open PDF file: {file_path}")
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return ""
            
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Try to open the document
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
        
        print(f"PDF has {len(document)} pages")
        
        text = ""
        for page_num in range(len(document)):
            print(f"Processing page {page_num+1}")
            page = document.load_page(page_num)
            page_text = page.get_text()
            print(f"Page {page_num+1} text length: {len(page_text)} characters")
            text += page_text
            
        document.close()
        print(f"Total extracted text length: {len(text)} characters")
        
        # Return empty if we got no text (might be a scanned PDF without OCR)
        if len(text.strip()) == 0:
            print("Warning: No text was extracted from the PDF, it might be a scanned document without OCR")
            
        return text
        
    except fitz.FileDataError as e:
        print(f"File data error: {e} - The file might be corrupted or not a valid PDF")
        return ""
    except fitz.EmptyFileError:
        print("The file is empty")
        return ""
    except PermissionError:
        print(f"Permission error: Cannot open file {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        traceback.print_exc()
        return ""

# Method 1: Try using PyPDF2 as an alternative
def read_pdf_with_pypdf2(file_path):
    """Alternative PDF reader using PyPDF2"""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
            return text
    except Exception as e:
        print(f"PyPDF2 error: {e}")
        traceback.print_exc()
        return ""

# Method 2: Try using pdfplumber as another alternative
def read_pdf_with_pdfplumber(file_path):
    """Alternative PDF reader using pdfplumber"""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"pdfplumber error: {e}")
        traceback.print_exc()
        return ""

# Method 3: As a last resort, try using pdf2image + pytesseract
def read_pdf_with_ocr(file_path):
    """Convert PDF to images and extract text using OCR"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        pages = convert_from_path(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    except Exception as e:
        print(f"OCR error: {e}")
        traceback.print_exc()
        return ""

def main():
    # Get resume file name via console input
    print("Enter the resume PDF filename (e.g., resume1.pdf):")
    resume_file = input().strip()
    
    # Check if the file exists
    if not os.path.exists(resume_file):
        print(f"Error: The file '{resume_file}' does not exist. Please check the filename and try again.")
        return
    
    # Try different methods to read the PDF
    resume = read_pdf(resume_file)
    
    if not resume:
        print("Trying alternative method 1 (PyPDF2)...")
        try:
            resume = read_pdf_with_pypdf2(resume_file)
        except ImportError:
            print("PyPDF2 not installed. Try 'pip install PyPDF2'")
    
    if not resume:
        print("Trying alternative method 2 (pdfplumber)...")
        try:
            resume = read_pdf_with_pdfplumber(resume_file)
        except ImportError:
            print("pdfplumber not installed. Try 'pip install pdfplumber'")
    
    if not resume:
        print("Trying alternative method 3 (OCR)...")
        try:
            resume = read_pdf_with_ocr(resume_file)
        except ImportError:
            print("OCR dependencies not installed. Try 'pip install pdf2image pytesseract'")
    
    if not resume:
        print("Could not read the resume PDF using any available method. Please check if it's a valid PDF file.")
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
