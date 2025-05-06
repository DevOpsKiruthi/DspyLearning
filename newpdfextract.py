import dspy
import json
import os
import subprocess
import traceback
from typing import List
from pydantic import BaseModel
import fitz  # PyMuPDF
from datetime import datetime

# Import Azure LLM configuration from config
from config import azure_llm

# Configure DSPy with Azure LLM
dspy.settings.configure(lm=azure_llm)

class AIReview(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    fit_score: int  # 0-100 scale
    comments: str

class CandidateEvaluation(BaseModel):
    name: str
    selected: bool
    not_selected: bool
    ai_review: AIReview

class ResumeEvaluationSignature(dspy.Signature):
    """Evaluate resume against job requirements"""
    resume_text: str = dspy.InputField(desc="Extracted resume content")
    job_description: str = dspy.InputField(desc="Detailed job requirements")
    additional_expectations: str = dspy.InputField(desc="Specific role expectations")
    
    evaluation: CandidateEvaluation = dspy.OutputField(
        desc="Structured evaluation with decision and analysis"
    )

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Robust PDF text extraction with multiple fallbacks"""
        try:
            # Normalize path for cross-platform compatibility
            pdf_path = os.path.normpath(pdf_path)
            
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                return ""
            
            # Try PyMuPDF first
            text = PDFProcessor._extract_with_pymupdf(pdf_path)
            if text and len(text.strip()) > 100:
                return text.strip()
            
            # Fallback methods
            methods = [
                PDFProcessor._extract_with_pypdf2,
                PDFProcessor._extract_with_pdfplumber,
                PDFProcessor._extract_with_ocr
            ]
            
            for method in methods:
                try:
                    text = method(pdf_path)
                    if text and len(text.strip()) > 100:
                        return text.strip()
                except Exception:
                    continue
            
            return ""
            
        except Exception as e:
            print(f"Error during text extraction: {str(e)}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _extract_with_pymupdf(pdf_path: str) -> str:
        """PyMuPDF implementation"""
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted and not doc.authenticate(""):
                return ""
            return "\n".join(page.get_text() for page in doc)

    @staticmethod
    def _extract_with_pypdf2(pdf_path: str) -> str:
        """PyPDF2 implementation"""
        from PyPDF2 import PdfReader
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> str:
        """pdfplumber implementation"""
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    @staticmethod
    def _extract_with_ocr(pdf_path: str) -> str:
        """OCR fallback for image-based PDFs"""
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(img) for img in images)

def get_multiline_input(prompt: str) -> str:
    """Collect multi-line input from user"""
    print(prompt)
    lines = []
    while True:
        try:
            line = input().strip()
            if not line:
                if lines:  # Only break if we have content
                    break
                continue
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    return "\n".join(lines)

def main():
    print("\nResume Evaluation System")
    print("=" * 40)
    
    # Get resume path
    while True:
        pdf_path = input("Enter path to resume PDF (or drag file here): ").strip('"\' ')
        if not pdf_path:
            print("Please provide a file path.")
            continue
            
        pdf_path = os.path.normpath(pdf_path)
        if os.path.exists(pdf_path):
            break
        print(f"File not found: {pdf_path}")
        print("Available files in current directory:")
        print("\n".join(f for f in os.listdir() if f.lower().endswith('.pdf')))
    
    # Extract text
    print("\nExtracting text...")
    resume_text = PDFProcessor.extract_text(pdf_path)
    
    if not resume_text:
        print("\nERROR: Failed to extract text. Possible issues:")
        print("- PDF is image-based/scanned (needs OCR)")
        print("- PDF is password protected")
        print("- PDF is corrupted")
        print("\nPlease check the file and try again.")
        return
    
    # Save extracted text
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = f"{base_name}_extracted.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(resume_text)
    print(f"Extracted text saved to: {txt_path}")
    
    # Get job requirements
    job_desc = get_multiline_input(
        "\nEnter job description (press Enter twice to finish):"
    )
    expectations = get_multiline_input(
        "\nEnter additional expectations (press Enter twice to finish):"
    )
    
    if not all([resume_text, job_desc, expectations]):
        print("Error: All inputs are required")
        return
    
    # Evaluate
    print("\nEvaluating candidate...")
    evaluator = dspy.ChainOfThought(ResumeEvaluationSignature)
    result = evaluator(
        resume_text=resume_text,
        job_description=job_desc,
        additional_expectations=expectations
    )
    
    # Process results
    evaluation = result.evaluation
    output = evaluation.model_dump_json(indent=2)
    print("\nEVALUATION RESULTS:")
    print(output)
    
    # Save JSON
    json_path = f"{base_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w') as f:
        f.write(output)
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
