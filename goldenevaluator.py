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
test_examples = [
    dspy.Example(
        resume="Software Engineer with 5 years experience in Python and JavaScript. Developed web applications using React and Django.",
        job_description="Looking for a Software Engineer with Python and React experience."
    ),
    dspy.Example(
        resume="Marketing specialist with experience in social media campaigns and content creation.",
        job_description="Looking for a Software Engineer with Python and React experience."
    ),
    dspy.Example(
        resume="Data Scientist with experience in machine learning, Python, and SQL.",
        job_description="Looking for a Data Scientist with machine learning and Python experience."
    )
]

# Add gold standard evaluations to each example
test_examples[0].gold_evaluation = {
    "name": "Candidate 1", 
    "selected": True, 
    "fit_score": 0.85
}

test_examples[1].gold_evaluation = {
    "name": "Candidate 2", 
    "selected": False, 
    "fit_score": 0.2
}

test_examples[2].gold_evaluation = {
    "name": "Candidate 3", 
    "selected": True, 
    "fit_score": 0.9
}

# Define a simple metric function
def resume_match_metric(gold, pred):
    """
    Evaluate how well the model's evaluation matches the gold standard
    Returns a dictionary of metrics
    """
    metrics = {}
    
    # Calculate selection accuracy (boolean match)
    metrics["selection_match"] = float(gold.gold_evaluation["selected"] == pred.selected)
    
    # Calculate fit score accuracy (how close the predicted score is to gold)
    score_diff = abs(gold.gold_evaluation["fit_score"] - pred.fit_score)
    metrics["fit_score_accuracy"] = 1.0 - score_diff  # Higher is better
    
    # Calculate an overall score (average of all metrics)
    metrics["overall"] = (metrics["selection_match"] + metrics["fit_score_accuracy"]) / 2
    
    return metrics

# Create the evaluator
evaluator = dspy.Evaluate(
    metric=resume_match_metric,
    num_threads=1  # Use single thread for simplicity
)

# Run the evaluation
if __name__ == "__main__":
    # Initialize the model
    model = SimpleResumeEvaluator()
    
    # Run evaluation
    results = evaluator(model, test_examples)
    
    # Print overall results
    print("\n----- EVALUATION RESULTS -----")
    print(f"Overall score: {results['overall']:.2f}")
    print(f"Selection match: {results['selection_match']:.2f}")
    print(f"Fit score accuracy: {results['fit_score_accuracy']:.2f}")
    
    # Print per-example results
    print("\n----- INDIVIDUAL EXAMPLES -----")
    for i, example in enumerate(results["examples"]):
        print(f"\nExample {i+1}:")
        print(f"Resume: {example['input'].resume[:50]}...")
        print(f"Job: {example['input'].job_description[:50]}...")
        print(f"Gold - Selected: {example['gold'].gold_evaluation['selected']}, Score: {example['gold'].gold_evaluation['fit_score']:.2f}")
        print(f"Pred - Selected: {example['pred'].selected}, Score: {example['pred'].fit_score:.2f}")
        print(f"Metrics - Selection: {example['selection_match']:.2f}, Score accuracy: {example['fit_score_accuracy']:.2f}")
