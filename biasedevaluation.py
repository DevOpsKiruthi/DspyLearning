def sentiment_analysis_metric(gold, pred):
    """
    Evaluate how well the model's sentiment analysis matches the gold standard
    Returns a percentage score (0-100)
    """
    # Calculate emotion category match
    emotion_match = 0.0
    gold_emotion = gold.gold_evaluation["emotion"].lower()
    pred_emotion = pred.emotion.lower()
    
    # Exact match
    if gold_emotion == pred_emotion:
        emotion_match = 1.0
    # Related emotions - count "angry" and "frustrated" as similar
    elif (gold_emotion == "angry" and pred_emotion == "frustrated") or \
         (gold_emotion == "frustrated" and pred_emotion == "angry"):
        emotion_match = 0.8
    # Related emotions - count "disappointed" and "sad" as similar
    elif (gold_emotion == "disappointed" and pred_emotion == "sad") or \
         (gold_emotion == "sad" and pred_emotion == "disappointed"):
        emotion_match = 0.8
    
    # Calculate intensity accuracy
    intensity_diff = abs(gold.gold_evaluation["intensity"] - pred.intensity)
    intensity_accuracy = max(0, 1.0 - intensity_diff)  # Higher is better, minimum 0
    
    # Calculate an overall score (weighted average of both metrics)
    overall = (emotion_match * 0.6) + (intensity_accuracy * 0.4)
    
    # Return the percentage (0-100)
    return overall * 100

# Create the evaluator with improved error handling
if __name__ == "__main__":
    # Initialize the model
    model = SentimentAnalysisEvaluator()
    
    # Create the evaluator with better error handling
    evaluator = dspy.Evaluate(
        metric=sentiment_analysis_metric,
        devset=test_examples,
        num_threads=1,
        display_progress=True,
        raise_exceptions=False  # Don't halt on errors
    )
    
    # Run evaluation
    try:
        results = evaluator(model)
        
        print("\n----- EVALUATION RESULTS -----")
        print(f"Overall score: {results:.2f}%")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    # Manually evaluate each example for detailed results with error handling
    print("\n----- INDIVIDUAL EXAMPLES -----")
    successful_examples = 0
    total_score = 0.0
    
    for i, example in enumerate(test_examples):
        try:
            # Run the model on the example
            prediction = model(sentence=example.sentence)
            
            # Calculate metrics
            gold_emotion = example.gold_evaluation["emotion"].lower()
            pred_emotion = prediction.emotion.lower()
            emotion_match = 1.0 if gold_emotion == pred_emotion else 0.0
            
            intensity_diff = abs(example.gold_evaluation["intensity"] - prediction.intensity)
            intensity_accuracy = max(0, 1.0 - intensity_diff)
            
            # Calculate overall score for this example
            example_score = ((emotion_match * 0.6) + (intensity_accuracy * 0.4)) * 100
            total_score += example_score
            successful_examples += 1
            
            # Print results
            print(f"\n✅ Example {i+1}:")
            print(f"Sentence: {example.sentence[:80]}...")
            print(f"Gold - Emotion: {example.gold_evaluation['emotion']}, " +
                  f"Intensity: {example.gold_evaluation['intensity']:.2f}")
            print(f"Pred - Emotion: {prediction.emotion}, " +
                  f"Intensity: {prediction.intensity:.2f}")
            print(f"Score: {example_score:.2f}%")
            
        except Exception as e:
            print(f"\n❌ Example {i+1}:")
            print(f"Sentence: {example.sentence[:80]}...")
            print(f"Error: {str(e)}")
    
    # Calculate and display the correct average score
    if successful_examples > 0:
        average_score = total_score / len(test_examples)  # Divide by TOTAL examples, not just successful ones
        print(f"\n----- FINAL RESULTS -----")
        print(f"Successful examples: {successful_examples}/{len(test_examples)}")
        print(f"Average score: {average_score:.2f}%")
    else:
        print("\n----- FINAL RESULTS -----")
        print("No examples were successfully evaluated.")
