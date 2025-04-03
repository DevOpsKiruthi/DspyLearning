import dspy
import json
import os
import subprocess
import traceback
from typing import List
from pydantic import BaseModel

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
        methods = [
            PDFProcessor._extract_with_pdftotext,
            PDFProcessor._extract_with_pymupdf,
            PDFProcessor._extract_with_pypdf2,
            PDFProcessor._extract_with_pdfplumber,
            PDFProcessor._extract_with_ocr
        ]
        
        for method in methods:
            try:
                text = method(pdf_path)
                if text and len(text.strip()) > 100:  # Minimum viable content
                    return text.strip()
            except Exception as e:
                print(f"Method {method.__name__} failed: {str(e)}")
                continue
        
        return ""

    @staticmethod
    def _extract_with_pdftotext(pdf_path: str) -> str:
        """System command (most reliable)"""
        try:
            result = subprocess.run(
                ['pdftotext', pdf_path, '-'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except Exception as e:
            print(f"pdftotext failed: {e}")
            return ""

    @staticmethod
    def _extract_with_pymupdf(pdf_path: str) -> str:
        """PyMuPDF implementation"""
        try:
            import fitz
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted and not doc.authenticate(""):
                    return ""
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"PyMuPDF failed: {e}")
            return ""

    @staticmethod
    def _extract_with_pypdf2(pdf_path: str) -> str:
        """PyPDF2 implementation"""
        try:
            from PyPDF2 import PdfReader
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            return ""

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> str:
        """pdfplumber implementation"""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            return ""

    @staticmethod
    def _extract_with_ocr(pdf_path: str) -> str:
        """OCR fallback for image-based PDFs"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(pdf_path)
            return "\n".join(pytesseract.image_to_string(img) for img in images)
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""

def get_interactive_input(prompt: str) -> str:
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
    pdf_path = input("Enter path to resume PDF: ").strip()
    print("\nExtracting text...")
    resume_text = PDFProcessor.extract_text(pdf_path)
    
    if not resume_text:
        print("\nERROR: Failed to extract text. Possible issues:")
        print("- File doesn't exist or path is incorrect")
        print("- PDF is image-based/scanned (needs OCR)")
        print("- PDF is password protected")
        print("- PDF is corrupted")
        print("\nPlease check the file and try again.")
        return
    
    # Save extracted text
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    with open(f"{base_name}_extracted.txt", 'w', encoding='utf-8') as f:
        f.write(resume_text)
    print(f"Extracted text saved to {base_name}_extracted.txt")
    
    # Get job requirements
    job_desc = get_interactive_input(
        "\nEnter job description (empty line to finish):"
    )
    expectations = get_interactive_input(
        "\nEnter additional expectations (empty line to finish):"
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
    with open(f"{base_name}_evaluation.json", 'w') as f:
        f.write(output)
    print(f"\nSaved to: {base_name}_evaluation.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
