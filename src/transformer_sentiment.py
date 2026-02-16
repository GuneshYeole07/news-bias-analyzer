from transformers import pipeline
import pandas as pd
from tqdm import tqdm

def analyze_with_transformers(text, max_length=512):
    """
    Use a pre-trained transformer model for sentiment
    This is more accurate than VADER!
    """
    
    # Load model (downloads automatically first time)
    print("Loading model... (this may take a minute the first time)")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Truncate if too long
    text = text[:max_length]
    
    # Get prediction
    result = sentiment_pipeline(text)[0]
    
    return {
        'label': result['label'],  # POSITIVE or NEGATIVE
        'score': result['score']    # Confidence 0-1
    }


def analyze_dataset_with_transformers(input_csv, output_csv):
    """
    Analyze entire dataset with transformers
    """
    
    df = pd.read_csv(input_csv)
    
    print(f"🤖 Analyzing {len(df)} articles with transformer model...")
    print("⚠️ This will take 5-10 minutes...")
    
    # Initialize model once
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    results = []
    
    # Process each article with progress bar
    for text in tqdm(df['text'].head(100)):  # Start with 100 articles
        try:
            result = sentiment_pipeline(str(text)[:512])[0]
            results.append({
                'transformer_label': result['label'],
                'transformer_score': result['score']
            })
        except:
            results.append({
                'transformer_label': 'UNKNOWN',
                'transformer_score': 0.0
            })
    
    # Add to dataframe
    results_df = pd.DataFrame(results)
    df_subset = df.head(100).copy()
    df_subset = pd.concat([df_subset.reset_index(drop=True), results_df], axis=1)
    
    # Save
    df_subset.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")
    
    # Compare with VADER
    print("\n📊 Comparison with VADER:")
    print(f"Transformer positive: {(results_df['transformer_label']=='POSITIVE').sum()}")
    print(f"VADER positive: {(df_subset['sentiment_label']=='positive').sum()}")
    
    return df_subset


if __name__ == "__main__":
    df = analyze_dataset_with_transformers(
        input_csv='data/processed/articles_with_sentiment.csv',
        output_csv='data/processed/articles_with_transformers.csv'
    )