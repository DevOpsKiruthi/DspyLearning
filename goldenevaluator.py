import dspy
from typing import List
from pydantic import BaseModel, Field

# Configure DSPy with your language model
from config import azure_llm
dspy.settings.configure(lm=azure_llm)

# Basic model for resume evaluation
class CandidateEvaluation(BaseModel):
    name: str
    selected: bool
    fit_score: float = Field(gt=0, lt=1)  # Ensures score is between 0 and 1

# Define the signature for resume evaluation
class ResumeEvaluationSignature(dspy.Signature):
    resume: str = dspy.InputField(desc="Candidate's resume text")
    job_description: str = dspy.InputField(desc="Job description")
    evaluation: CandidateEvaluation = dspy.OutputField(desc="Evaluation of the candidate")

# Create a simple module that evaluates resumes
class SimpleResumeEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(ResumeEvaluationSignature)
    
    def forward(self, resume, job_description):
        result = self.evaluate(resume=resume, job_description=job_description)
        return result.evaluation

# Create a few example resumes and job descriptions for testing
# FIX: Specify which fields are inputs using with_inputs()
test_examples = [
    dspy.Example(
        resume="Software Engineer with 5 years experience in Python and JavaScript. Developed web applications using React and Django.",
        job_description="Looking for a Software Engineer with Python and React experience.",
        gold_evaluation={
            "name": "Candidate 1", 
            "selected": True, 
            "fit_score": 0.85
        }
    ).with_inputs("resume", "job_description"),
    
    dspy.Example(
        resume="Marketing specialist with experience in social media campaigns and content creation.",
        job_description="Looking for a Software Engineer with Python and React experience.",
        gold_evaluation={
            "name": "Candidate 2", 
            "selected": False, 
            "fit_score": 0.2
        }
    ).with_inputs("resume", "job_description"),
    
    dspy.Example(
        resume="Data Scientist with experience in machine learning, Python, and SQL.",
        job_description="Looking for a Data Scientist with machine learning and Python experience.",
        gold_evaluation={
            "name": "Candidate 3", 
            "selected": True, 
            "fit_score": 0.9
        }
    ).with_inputs("resume", "job_description")
]

# Define a simple metric function
def resume_match_metric(gold, pred):
    """
    Evaluate how well the model's evaluation matches the gold standard
    Returns a dictionary of metrics
    """
    # Calculate selection accuracy (boolean match)
    selection_match = float(gold.gold_evaluation["selected"] == pred.selected)
    
    # Calculate fit score accuracy (how close the predicted score is to gold)
    score_diff = abs(gold.gold_evaluation["fit_score"] - pred.fit_score)
    fit_score_accuracy = 1.0 - score_diff  # Higher is better
    
    # Calculate an overall score (average of all metrics)
    overall = (selection_match + fit_score_accuracy) / 2
    
    # FIX: Return a single value (overall score), not a dictionary
    return overall

# Run the evaluation
if __name__ == "__main__":
    # Initialize the model
    model = SimpleResumeEvaluator()
    
    # Create the evaluator
    evaluator = dspy.Evaluate(
        metric=resume_match_metric,
        devset=test_examples,
        num_threads=1
    )
    
    # Run evaluation
    results = evaluator(model)
    
    # FIX: Handle results as a float, not a dictionary
    print("\n----- EVALUATION RESULTS -----")
    print(f"Overall score: {results:.2f}")
    
    # If you need more detailed metrics, you'll need to calculate them separately
    # or adjust the evaluate function to store them somewhere
    
    # For demonstration, let's manually run the evaluation on each example
    # to get detailed results
    print("\n----- INDIVIDUAL EXAMPLES -----")
    for i, example in enumerate(test_examples):
        # Run the model on the example
        try:
            prediction = model(resume=example.resume, job_description=example.job_description)
            
            # Calculate metrics
            selection_match = float(example.gold_evaluation["selected"] == prediction.selected)
            score_diff = abs(example.gold_evaluation["fit_score"] - prediction.fit_score)
            fit_score_accuracy = 1.0 - score_diff
            
            # Print results
            print(f"\nExample {i+1}:")
            print(f"Resume: {example.resume[:50]}...")
            print(f"Job: {example.job_description[:50]}...")
            print(f"Gold - Selected: {example.gold_evaluation['selected']}, Score: {example.gold_evaluation['fit_score']:.2f}")
            print(f"Pred - Selected: {prediction.selected}, Score: {prediction.fit_score:.2f}")
            print(f"Metrics - Selection: {selection_match:.2f}, Score accuracy: {fit_score_accuracy:.2f}")
        except Exception as e:
            print(f"\nExample {i+1} - Error: {str(e)}")
