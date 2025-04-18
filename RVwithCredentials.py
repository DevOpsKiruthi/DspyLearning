import streamlit as st
import os
import dspy
import tempfile
import traceback
from typing import List
from pydantic import BaseModel
from pdf2image import convert_from_path
import pytesseract
from config import azure_llm  # Make sure this file/config exists and is correct

# ✅ SET THIS IMMEDIATELY AFTER IMPORTS
st.set_page_config(page_title="Resume Evaluator AI", layout="centered")

# -----------------------
# 1. USER LOGIN AUTH
# -----------------------

# Dummy users (username: password)
users = {
    "admin": "#6bwBcoe&ZuxH4dH38d",
    "demo": "#6bwBcoe&ZuxH4dH38d"
}

def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

def logout():
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())

# Login logic
if "authenticated" not in st.session_state:
    login()
    st.stop()
else:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    logout()

# -----------------------
# 2. RESUME EVALUATOR
# -----------------------

# Configure DSPy
dspy.settings.configure(lm=azure_llm)

# Pydantic models
class AIReview(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    fit_score: float
    comments: str

class CandidateEvaluation(BaseModel):
    name: str
    selected: bool
    ai_review: AIReview

# DSPy Signature
class ResumeEvaluationSignature(dspy.Signature):
    resume: str = dspy.InputField(desc="Candidate's full resume text")
    job_description: str = dspy.InputField(desc="Complete job description")
    what_we_are_expecting: str = dspy.InputField(desc="Additional comments or expectations for the role")
    evaluation: CandidateEvaluation = dspy.OutputField(desc="Structured evaluation output")

resume_evaluator = dspy.ChainOfThought(ResumeEvaluationSignature)

# OCR PDF text extractor
def extract_text_from_pdf(file_path):
    try:
        images = convert_from_path(file_path)
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        traceback.print_exc()
        return ""

# Streamlit UI
st.title("🤖 AI Resume Evaluator")
st.write("Upload a candidate resume and evaluate it against a job description.")

with st.form("evaluation_form"):
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("🧾 Job Description", height=150)
    expectations = st.text_area("🎯 Additional Expectations", height=100)
    submitted = st.form_submit_button("🔍 Evaluate")

if submitted:
    if not resume_file or not job_description or not expectations:
        st.warning("Please fill in all fields.")
    else:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(resume_file.read())
            resume_path = tmp_file.name

        st.info("Extracting resume text using OCR...")
        resume_text = extract_text_from_pdf(resume_path)

        if not resume_text.strip():
            st.error("Failed to extract text from the PDF.")
        else:
            st.info("Running evaluation...")
            try:
                result = resume_evaluator(
                    resume=resume_text,
                    job_description=job_description,
                    what_we_are_expecting=expectations
                )
                evaluation = result.evaluation

                st.success("✅ Evaluation Complete")
                st.subheader("📋 Candidate Evaluation")
                st.write(f"**Name:** {evaluation.name}")
                st.write(f"**Selected:** {'✅ Yes' if evaluation.selected else '❌ No'}")
                st.write(f"**Fit Score:** {evaluation.ai_review.fit_score:.2f}")
                st.write(f"**Strengths:**")
                st.markdown("\n".join(f"- {s}" for s in evaluation.ai_review.strengths))
                st.write(f"**Weaknesses:**")
                st.markdown("\n".join(f"- {w}" for w in evaluation.ai_review.weaknesses))
                st.write(f"**Comments:** {evaluation.ai_review.comments}")

                with st.expander("🛠️ DSPy Process History"):
                    dspy.inspect_history(n=1)

            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                traceback.print_exc()
