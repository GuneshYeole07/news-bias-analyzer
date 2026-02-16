from openai import OpenAI
import pandas as pd
import time

def analyze_bias_with_llm(article_text, api_key):
    """
    Use GPT to analyze bias - much more sophisticated!
    
    Cost: ~$0.001 per article
    """
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Analyze the political bias of this news article.
    
    Article:
    {article_text[:1000]}
    
    Provide:
    1. Bias classification: left, center-left, center, center-right, or right
    2. Confidence: low, medium, or high
    3. One sentence explaining your reasoning
    
    Format your response as:
    Bias: [classification]
    Confidence: [confidence]
    Reasoning: [one sentence]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Cheapest model
            messages=[
                {"role": "system", "content": "You are a media bias expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        return result
        
    except Exception as e:
        return f"Error: {e}"


def analyze_sample_with_llm(input_csv, api_key, num_samples=10):
    """
    Analyze a few articles with LLM (to save money)
    """
    
    df = pd.read_csv(input_csv)
    
    print(f"🤖 Analyzing {num_samples} sample articles with GPT...")
    
    results = []
    
    for i in range(min(num_samples, len(df))):
        print(f"\nAnalyzing article {i+1}/{num_samples}...")
        
        text = df.iloc[i]['text']
        result = analyze_bias_with_llm(text, api_key)
        
        print(result)
        results.append(result)
        
        # Be nice to API (wait 1 second)
        time.sleep(1)
    
    # Save results
    df_sample = df.head(num_samples).copy()
    df_sample['llm_analysis'] = results
    
    df_sample.to_csv('data/processed/llm_analyzed.csv', index=False)
    print(f"\n✅ Saved to data/processed/llm_analyzed.csv")
    
    return df_sample


if __name__ == "__main__":
    # Get your API key from https://platform.openai.com/api-keys
    API_KEY = "sk-..."  # Replace with your key
    
    df = analyze_sample_with_llm(
        input_csv='data/processed/articles_with_sentiment.csv',
        api_key=API_KEY,
        num_samples=5  # Start small!
    )