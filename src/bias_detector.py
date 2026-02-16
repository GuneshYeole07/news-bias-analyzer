import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

def create_bias_labels(df):
    """
    Manually label sources with known political bias
    
    This is a simplified version - in reality, you'd use AllSides ratings
    or a labeled dataset
    """
    
    # Example bias labels (replace with actual research)
    bias_map = {
        'CNN': 'left',
        'MSNBC': 'left',
        'NPR': 'center-left',
        'BBC News': 'center',
        'Reuters': 'center',
        'The Hill': 'center',
        'Fox News': 'right',
        'New York Post': 'right',
        'Washington Examiner': 'right'
    }
    
    # Map sources to bias labels
    df['bias_label'] = df['source'].map(bias_map)
    
    # Remove unlabeled sources
    df_labeled = df.dropna(subset=['bias_label'])
    
    print(f"📊 Labeled {len(df_labeled)}/{len(df)} articles")
    print(f"\nBias distribution:")
    print(df_labeled['bias_label'].value_counts())
    
    return df_labeled


def train_bias_classifier(df):
    """
    Train a simple ML model to detect bias
    """
    
    print("🤖 Training bias classifier...")
    
    # Prepare data
    X = df['text_clean']
    y = df['bias_label']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} articles")
    print(f"Test set: {len(X_test)} articles")
    
    # Convert text to numbers (TF-IDF)
    print("\n📝 Converting text to features...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("🎯 Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model accuracy: {accuracy:.2%}")
    print("\n📊 Detailed results:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('models/bias_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("💾 Model saved to models/")
    
    return model, vectorizer


def predict_bias(text, model, vectorizer):
    """
    Predict bias of a new article
    """
    
    # Clean and vectorize
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    return {
        'prediction': prediction,
        'confidence': max(probability)
    }


if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/processed/articles_with_sentiment.csv')
    
    # Add bias labels
    df_labeled = create_bias_labels(df)
    
    # Train model
    model, vectorizer = train_bias_classifier(df_labeled)
    
    # Test on a sample
    print("\n🧪 Testing on a sample article:")
    sample_text = df_labeled.iloc[0]['text_clean']
    result = predict_bias(sample_text, model, vectorizer)
    print(f"Predicted bias: {result['prediction']} (confidence: {result['confidence']:.2%})")