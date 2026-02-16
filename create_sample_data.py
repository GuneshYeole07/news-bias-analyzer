import pandas as pd
import numpy as np
import os

# Create sample data for demo
np.random.seed(42)

sources = ['BBC News', 'CNN', 'Reuters', 'The Guardian', 'NPR', 
           'TechCrunch', 'The Verge', 'Wired', 'Fox News', 'MSNBC']

sentiments = ['positive', 'neutral', 'negative']

titles = [
    "New AI breakthrough announced by researchers",
    "Climate change talks reach historic agreement",
    "Stock market shows mixed signals amid uncertainty",
    "Tech giant announces major product launch",
    "Scientists discover potential cancer treatment",
    "Political tensions rise over trade negotiations",
    "Renewable energy costs continue to fall",
    "Space exploration mission achieves milestone",
    "Economic indicators suggest steady growth",
    "Social media platform faces privacy concerns"
] * 25  # 250 articles

# Generate sample data
data = []
for i in range(250):
    sentiment = np.random.choice(sentiments, p=[0.68, 0.09, 0.23])
    
    if sentiment == 'positive':
        compound = np.random.uniform(0.05, 0.9)
    elif sentiment == 'negative':
        compound = np.random.uniform(-0.9, -0.05)
    else:
        compound = np.random.uniform(-0.05, 0.05)
    
    data.append({
        'title': titles[i],
        'text': f"This is sample article text about {titles[i].lower()}. " * 10,
        'text_clean': f"sample article text {titles[i].lower()}",
        'source': np.random.choice(sources),
        'url': f"https://example.com/article-{i}",
        'published_at': '2025-01-15',
        'date_collected': '2025-01-16',
        'word_count': np.random.randint(40, 150),
        'sentiment_compound': compound,
        'sentiment_label': sentiment,
        'sentiment_pos': np.random.uniform(0, 0.5),
        'sentiment_neg': np.random.uniform(0, 0.3),
        'sentiment_neu': np.random.uniform(0.5, 0.8)
    })

df = pd.DataFrame(data)

# Create directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Save to CSV
df.to_csv('data/processed/articles_with_sentiment.csv', index=False)

print(f"✅ Created sample data: {len(df)} articles")
print(f"📁 Saved to: data/processed/articles_with_sentiment.csv")
print(f"\nSentiment distribution:")
print(df['sentiment_label'].value_counts())
