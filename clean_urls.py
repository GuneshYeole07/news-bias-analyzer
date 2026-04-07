import pandas as pd
import os

print("🔧 Cleaning URLs in your data...")

# Load the data
df = pd.read_csv('data/processed/articles_with_sentiment.csv')

print(f"📊 Total articles before cleaning: {len(df)}")

# Check what URLs look like
print("\n📋 Sample URLs:")
print(df['url'].head(10))

# Remove invalid URLs
invalid_patterns = ['example.com', 'claude.ai', 'removed', 'localhost', 'test.com']

# Create a mask for valid URLs
valid_mask = df['url'].notna()  # Not null
valid_mask &= df['url'].str.startswith('http')  # Starts with http
for pattern in invalid_patterns:
    valid_mask &= ~df['url'].str.contains(pattern, case=False, na=False)

# Filter to only valid articles
df_clean = df[valid_mask].copy()

print(f"\n✅ Articles after cleaning: {len(df_clean)}")
print(f"❌ Removed {len(df) - len(df_clean)} articles with invalid URLs")

if len(df_clean) > 0:
    # Save cleaned data
    df_clean.to_csv('data/processed/articles_with_sentiment.csv', index=False)
    print("\n✅ Data cleaned and saved!")
    print("\n📋 Sample clean URLs:")
    print(df_clean['url'].head(5))
else:
    print("\n⚠️ WARNING: No valid URLs found!")
    print("You need to re-run collect_real_data.py to get fresh articles with valid URLs")