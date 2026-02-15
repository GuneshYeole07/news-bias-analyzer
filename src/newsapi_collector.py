import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def collect_news(api_key, topic='technology', num_articles=100):
    """
    Collect news articles using NewsAPI
    """
    
    print(f"🔍 Searching for articles about '{topic}'...")
    
    # NewsAPI endpoint
    url = 'https://newsapi.org/v2/everything'
    
    # Date range (last 30 days)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30)
    
    # Parameters
    params = {
        'q': topic,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': min(num_articles, 100),
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d')
    }
    
    # Make request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        
        print(f"✅ Found {len(articles)} articles!")
        
        # Format data
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                'title': article.get('title', ''),
                'text': (article.get('description', '') or '') + ' ' + (article.get('content', '') or ''),
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'author': article.get('author', ''),
                'date_collected': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Create DataFrame
        df = pd.DataFrame(formatted_articles)
        
        # Remove rows with empty text
        df = df[df['text'].str.len() > 50]
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        filename = f'data/raw/articles_{topic.replace(" ", "_")}.csv'
        df.to_csv(filename, index=False)
        
        print(f"💾 Saved {len(df)} articles to {filename}")
        print(f"\n📊 Sources found:")
        print(df['source'].value_counts().head(10))
        
        return df
        
    else:
        print(f"❌ Error {response.status_code}: {response.json().get('message', 'Unknown error')}")
        return None


# Main execution
if __name__ == "__main__":
    
    # 🔑 PASTE YOUR API KEY HERE (between the quotes)
    API_KEY = "0354dc1957e349449119c91b7328af36"
    
    print("📰 News Article Collector")
    print("=" * 50)
    
    # Collect articles on multiple topics
    topics = [
        'technology',
        'sports', 
        'politics'
    ]
    
    all_data = []
    
    for topic in topics:
        print(f"\n--- Collecting: {topic} ---")
        df = collect_news(API_KEY, topic=topic, num_articles=100)
        
        if df is not None:
            all_data.append(df)
    
    # Combine all articles
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['title', 'source'])
        
        # Save combined file
        combined.to_csv('data/raw/all_articles.csv', index=False)
        
        print("\n" + "=" * 50)
        print(f"🎉 SUCCESS!")
        print(f"📊 Total articles: {len(combined)}")
        print(f"📰 Unique sources: {combined['source'].nunique()}")
        print(f"💾 Saved to: data/raw/all_articles.csv")
        
        print(f"\n📈 Top sources:")
        print(combined['source'].value_counts().head(10))
    else:
        print("\n❌ No articles collected. Check your API key!")