import streamlit as st
import os
import dspy
import tempfile
import traceback
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from pdf2image import convert_from_path
import pytesseract
from config import azure_llm

# ✅ SET THIS IMMEDIATELY AFTER IMPORTS
st.set_page_config(page_title="Bulk Resume Evaluator AI", layout="wide")

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

# Flattened Pydantic model with Field validation
class CandidateEvaluation(BaseModel):
    name: str
    selected: bool
    strengths: List[str]
    weaknesses: List[str]
    fit_score: float = Field(gt=0, lt=1)  # Enforces 0 < fit_score < 1
    comments: str

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
st.title("🤖 Bulk AI Resume Evaluator")
st.write("Upload multiple candidate resumes and evaluate them against a job description.")

with st.form("evaluation_form"):
    resume_files = st.file_uploader("📄 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("🧾 Job Description", height=150)
    expectations = st.text_area("🎯 Additional Expectations", height=100)
    submitted = st.form_submit_button("🔍 Evaluate All Resumes")

if submitted:
    if not resume_files or not job_description or not expectations:
        st.warning("Please fill in all fields and upload at least one resume.")
    else:
        # Initialize results container
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each resume
        for i, resume_file in enumerate(resume_files):
            status_text.text(f"Processing {i+1}/{len(resume_files)}: {resume_file.name}")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(resume_file.read())
                resume_path = tmp_file.name

            resume_text = extract_text_from_pdf(resume_path)

            if not resume_text.strip():
                results.append({
                    "File": resume_file.name,
                    "Status": "Failed to extract text",
                    "Name": "N/A",
                    "Selected": False,
                    "Fit Score": 0,
                    "Strengths": [],
                    "Weaknesses": [],
                    "Comments": "Failed to process this resume"
                })
            else:
                try:
                    result = resume_evaluator(
                        resume=resume_text,
                        job_description=job_description,
                        what_we_are_expecting=expectations
                    )
                    evaluation = result.evaluation
                    
                    # Append to results
                    results.append({
                        "File": resume_file.name,
                        "Status": "Processed",
                        "Name": evaluation.name,
                        "Selected": evaluation.selected,
                        "Fit Score": evaluation.fit_score,
                        "Strengths": evaluation.strengths,
                        "Weaknesses": evaluation.weaknesses,
                        "Comments": evaluation.comments
                    })
                    
                except Exception as e:
                    results.append({
                        "File": resume_file.name,
                        "Status": f"Error: {str(e)}",
                        "Name": "Error",
                        "Selected": False,
                        "Fit Score": 0,
                        "Strengths": [],
                        "Weaknesses": [],
                        "Comments": f"Processing error: {str(e)}"
                    })
            
            # Clean up temp file
            try:
                os.unlink(resume_path)
            except:
                pass
                
            # Update progress
            progress_bar.progress((i + 1) / len(resume_files))
            
        # Display results
        status_text.text("Evaluation complete! Displaying results...")
        
        # Convert results to DataFrame for better display
        df = pd.DataFrame(results)
        
        # Top candidates section
        st.subheader("🏆 Top Candidates")
        selected_candidates = df[df["Selected"] == True].sort_values("Fit Score", ascending=False)
        if not selected_candidates.empty:
            for i, row in selected_candidates.iterrows():
                with st.expander(f"✅ {row['Name']} - Fit Score: {row['Fit Score']:.2f}"):
                    st.write(f"**File:** {row['File']}")
                    st.write(f"**Strengths:**")
                    if isinstance(row['Strengths'], list):
                        st.markdown("\n".join(f"- {s}" for s in row['Strengths']))
                    st.write(f"**Weaknesses:**")
                    if isinstance(row['Weaknesses'], list):
                        st.markdown("\n".join(f"- {w}" for w in row['Weaknesses']))
                    st.write(f"**Comments:** {row['Comments']}")
        else:
            st.write("No candidates were selected as suitable for this position.")
        
        # All candidates
        st.subheader("📊 All Candidates")
        
        # Add a download button for CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "resume_evaluation_results.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Display table with all candidates
        filtered_df = df[["File", "Name", "Selected", "Fit Score", "Status", "Comments"]]
        st.dataframe(filtered_df)
