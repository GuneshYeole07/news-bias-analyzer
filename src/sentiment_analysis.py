import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download VADER lexicon (run once)
import nltk
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(text):
    """
    Analyze sentiment of text
    
    Returns:
    - Compound score: -1 (very negative) to +1 (very positive)
    - Label: 'positive', 'negative', or 'neutral'
    """
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    # Compound score
    compound = scores['compound']
    
    # Determine label
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'compound': compound,
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'label': label
    }


def analyze_dataset_sentiment(input_csv, output_csv):
    """
    Add sentiment scores to dataset
    """
    
    print(f"📥 Loading data...")
    df = pd.read_csv(input_csv)
    
    print(f"🎭 Analyzing sentiment for {len(df)} articles...")
    
    # Analyze each article
    sentiment_results = df['text'].apply(analyze_sentiment)
    
    # Add to dataframe
    df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
    df['sentiment_label'] = sentiment_results.apply(lambda x: x['label'])
    df['sentiment_pos'] = sentiment_results.apply(lambda x: x['positive'])
    df['sentiment_neg'] = sentiment_results.apply(lambda x: x['negative'])
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv}")
    
    # Print statistics
    print(f"\n📊 Sentiment Distribution:")
    print(df['sentiment_label'].value_counts())
    
    print(f"\nAverage sentiment by source:")
    print(df.groupby('source')['sentiment_compound'].mean().sort_values(ascending=False))
    
    return df


def visualize_sentiment(df):
    """
    Create sentiment visualizations
    """
    
    # Sentiment distribution
    plt.figure(figsize=(10, 6))
    df['sentiment_label'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_distribution.png')
    plt.show()
    
    # Sentiment by source
    plt.figure(figsize=(12, 6))
    sentiment_by_source = df.groupby('source')['sentiment_compound'].mean().sort_values()
    sentiment_by_source.plot(kind='barh', color='skyblue')
    plt.title('Average Sentiment by Source')
    plt.xlabel('Sentiment Score (-1 to +1)')
    plt.ylabel('Source')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_by_source.png')
    plt.show()


if __name__ == "__main__":
    # Analyze sentiment
    df = analyze_dataset_sentiment(
        input_csv='data/processed/articles_clean.csv',
        output_csv='data/processed/articles_with_sentiment.csv'
    )
    
    # Create visualizations
    visualize_sentiment(df)
    
    print("\n🎉 Sentiment analysis complete!")