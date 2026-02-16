import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """
    Clean and normalize text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text):
    """
    Remove common words like 'the', 'is', 'and'
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    
    # Keep only meaningful words
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)


def preprocess_dataset(input_csv, output_csv):
    """
    Preprocess entire dataset
    """
    print(f"📥 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"🧹 Cleaning {len(df)} articles...")
    
    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Remove stopwords
    print("🔤 Removing stopwords...")
    df['text_no_stopwords'] = df['text_clean'].apply(remove_stopwords)
    
    # Calculate word count
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save processed data
    print(f"💾 Saving to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Done! Processed {len(df)} articles")
    print(f"\nAverage word count: {df['word_count'].mean():.0f} words")
    print(f"\nSample cleaned text:")
    print(df.iloc[0]['text_clean'][:200] + "...")
    
    return df


if __name__ == "__main__":
    # Process your data
    df = preprocess_dataset(
        input_csv='data/raw/all_articles.csv',
        output_csv='data/processed/articles_clean.csv'
    )