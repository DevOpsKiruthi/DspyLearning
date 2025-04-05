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

def read_pdf_with_ocr(file_path):
    """Convert PDF to images and extract text using OCR with improved configuration"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        print(f"Starting OCR processing for: {file_path}")
        
        # Convert PDF to images with higher DPI for better OCR results
        pages = convert_from_path(file_path, dpi=300)
        print(f"Converted PDF to {len(pages)} images")
        
        # Process each page with OCR
        text = ""
        for i, page in enumerate(pages):
            print(f"Processing page {i+1} with OCR...")
            # Use better OCR configuration
            page_text = pytesseract.image_to_string(
                page,
                config='--psm 6'  # Assume a single uniform block of text
            )
            text += page_text + "\n"
            print(f"Extracted {len(page_text)} characters from page {i+1}")
            
        print(f"Total OCR text extracted: {len(text)} characters")
        return text
    except ImportError as e:
        print(f"OCR dependencies missing: {e}")
        print("Please install with: pip install pdf2image pytesseract")
        print("And system packages: sudo apt install poppler-utils tesseract-ocr")
        return ""
    except Exception as e:
        print(f"OCR error: {e}")
        traceback.print_exc()
        return ""

def read_pdf_with_pymupdf(file_path):
    """Extract text directly from PDF using PyMuPDF"""
    try:
        import fitz  # PyMuPDF
        
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
            
        print(f"Successfully extracted {len(text)} characters with PyMuPDF")
        return text
        
    except ImportError:
        print("PyMuPDF not installed. Try 'pip install pymupdf'")
        return ""
    except Exception as e:
        print(f"PyMuPDF error: {e}")
        traceback.print_exc()
        return ""

def read_pdf_with_pypdf2(file_path):
    """Extract text using PyPDF2 as a fallback method"""
    try:
        import PyPDF2
        
        print(f"Attempting text extraction with PyPDF2: {file_path}")
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if encrypted
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                    print("Decrypted PDF with empty password")
                except:
                    print("Could not decrypt the PDF")
                    return ""
            
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
                
        print(f"Extracted {len(text)} characters with PyPDF2")
        return text
        
    except ImportError:
        print("PyPDF2 not installed. Try 'pip install PyPDF2'")
        return ""
    except Exception as e:
        print(f"PyPDF2 error: {e}")
        traceback.print_exc()
        return ""

def read_pdf_with_pdfplumber(file_path):
    """Extract text using pdfplumber as another fallback method"""
    try:
        import pdfplumber
        
        print(f"Attempting text extraction with pdfplumber: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                
        print(f"Extracted {len(text)} characters with pdfplumber")
        return text
        
    except ImportError:
        print("pdfplumber not installed. Try 'pip install pdfplumber'")
        return ""
    except Exception as e:
        print(f"pdfplumber error: {e}")
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
    
    # First, try OCR directly since we're prioritizing for scanned documents
    print("Attempting OCR extraction first...")
    resume = ""
    try:
        resume = read_pdf_with_ocr(resume_file)
    except ImportError:
        print("OCR dependencies not installed.")
        print("Installing required packages for OCR...")
        print("Please run the following commands:")
        print("pip install pdf2image pytesseract")
        print("For Linux: sudo apt install poppler-utils tesseract-ocr")
    
    # If OCR failed or returned minimal text, try other methods
    if not resume or len(resume.strip()) < 100:
        print("OCR extraction failed or returned minimal text. Trying standard methods...")
        
        # Try PyMuPDF first as it's usually more reliable
        resume = read_pdf_with_pymupdf(resume_file)
        
        if not resume or len(resume.strip()) < 100:
            print("PyMuPDF extraction failed or returned minimal text. Trying PyPDF2...")
            try:
                resume = read_pdf_with_pypdf2(resume_file)
            except ImportError:
                print("PyPDF2 not installed. Try 'pip install PyPDF2'")
        
        if not resume or len(resume.strip()) < 100:
            print("PyPDF2 extraction failed or returned minimal text. Trying pdfplumber...")
            try:
                resume = read_pdf_with_pdfplumber(resume_file)
            except ImportError:
                print("pdfplumber not installed. Try 'pip install pdfplumber'")
    
    if not resume:
        print("Could not extract text from the PDF using any available method.")
        return
    
    if len(resume.strip()) < 100:
        print("Warning: Extracted text is very short. This may indicate a poor quality scan or image-based PDF.")
        print(f"Only extracted {len(resume.strip())} characters.")
    
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
    try:
        result = resume_evaluator(
            resume=resume,
            job_description=job_description,
            what_we_are_expecting=expectations
        )
        
        # Print the structured output
        evaluation = result.evaluation
        output = evaluation.model_dump_json(indent=2)
        print
