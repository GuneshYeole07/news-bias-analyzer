from transformers import pipeline
import pandas as pd

def analyze_with_zero_shot(text):
    """
    Use free zero-shot classification
    No training needed!
    """
    
    print("Loading model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    
    # Define bias categories
    candidate_labels = [
        "left-leaning political bias",
        "center political bias",
        "right-leaning political bias"
    ]
    
    # Classify
    result = classifier(text[:1000], candidate_labels)
    
    # Get top prediction
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    return {
        'bias': top_label,
        'confidence': top_score
    }


if __name__ == "__main__":
    # Test on one article
    df = pd.read_csv('data/processed/articles_clean.csv')
    
    sample_text = df.iloc[0]['text']
    result = analyze_with_zero_shot(sample_text)
    
    print(f"Predicted bias: {result['bias']}")
    print(f"Confidence: {result['confidence']:.2%}")
