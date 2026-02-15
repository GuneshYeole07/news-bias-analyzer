import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

def scrape_single_article(url):
    """
    Scrape one article from a URL
    
    Parameters:
    url (str): The webpage URL
    
    Returns:
    dict: Article title, text, and source
    """
    
    try:
        # Step 1: Download the webpage
        print(f"📥 Downloading: {url}")
        response = requests.get(url, timeout=10)
        
        # Step 2: Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Step 3: Extract title (this works for most news sites)
        title = soup.find('h1')
        if title:
            title = title.get_text().strip()
        else:
            title = "No title found"
        
        # Step 4: Extract article text
        # Look for paragraph tags
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Step 5: Get the source (domain name)
        source = url.split('/')[2]  # Extract domain from URL
        
        # Step 6: Return as dictionary
        return {
            'title': title,
            'text': article_text[:1000],  # First 1000 characters
            'source': source,
            'url': url,
            'date_scraped': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None


# Test it with one article
if __name__ == "__main__":
    # Example URL (replace with any news article)
    test_url = "https://www.bbc.com/news/world-us-canada-68259864"
    
    article = scrape_single_article(test_url)
    
    if article:
        print("\n✅ Successfully scraped!")
        print(f"Title: {article['title']}")
        print(f"Text preview: {article['text'][:200]}...")
        print(f"Source: {article['source']}")