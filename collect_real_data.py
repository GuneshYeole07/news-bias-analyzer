import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def collect_real_articles(api_key, topics, articles_per_topic=50):
    """
    Collect REAL articles from NewsAPI with REAL URLs
    """
    
    all_articles = []
    
    for topic in topics:
        print(f"\n🔍 Collecting articles about: {topic}")
        
        url = 'https://newsapi.org/v2/everything'
        
        # Get articles from last 7 days
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        params = {
            'q': topic,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': articles_per_topic,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            for article in articles:
                # Skip articles without proper URLs
                if not article.get('url') or 'removed' in article.get('url', '').lower():
                    continue
                
                all_articles.append({
                    'title': article.get('title', 'No title'),
                    'text': (article.get('description', '') or '') + ' ' + (article.get('content', '') or ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),  # REAL URL!
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'date_collected': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            print(f"✅ Collected {len(articles)} articles for '{topic}'")
        else:
            print(f"❌ Error fetching '{topic}': {response.status_code}")
    
    return all_articles


def main():
    # 🔑 PUT YOUR API KEY HERE
    API_KEY = "0354dc1957e349449119c91b7328af36"
    
    # Topics to search for
    topics = [
        'artificial intelligence',
        'climate change',
        'technology',
        'politics',
        'economy'
    ]
    
    print("📰 Collecting REAL news articles...")
    print("=" * 50)
    
    # Collect articles
    articles = collect_real_articles(API_KEY, topics, articles_per_topic=30)
    
    if articles:
        # Create DataFrame
        df = pd.DataFrame(articles)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'url'])
        
        # Add word count
        df['word_count'] = df['text'].str.split().str.len()
        
        # Filter out very short articles
        df = df[df['word_count'] > 20]
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/real_articles.csv', index=False)
        
        print("\n" + "=" * 50)
        print(f"🎉 SUCCESS!")
        print(f"📊 Total articles: {len(df)}")
        print(f"📰 Unique sources: {df['source'].nunique()}")
        print(f"💾 Saved to: data/raw/real_articles.csv")
        
        print(f"\n📈 Top sources:")
        print(df['source'].value_counts().head(10))
        
        print(f"\n🔗 Sample real URLs:")
        for url in df['url'].head(3):
            print(f"  • {url}")
    else:
        print("\n❌ No articles collected!")


if __name__ == "__main__":
    main()