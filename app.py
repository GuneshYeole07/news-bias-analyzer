import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="News Bias Analyzer 📰",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR MODERN LOOK
# ============================================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card-like containers */
    .css-1r6slb0 {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Headers */
    h1 {
        color: white;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Select boxes */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 10px;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/articles_with_sentiment.csv')
        return df
    except FileNotFoundError:
        return None

# ============================================
# HELPER FUNCTIONS
# ============================================
def create_sentiment_donut(df):
    """Create an attractive donut chart for sentiment"""
    sentiment_counts = df['sentiment_label'].value_counts()
    
    colors = {
        'positive': '#10b981',
        'neutral': '#6b7280',
        'negative': '#ef4444'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.6,
        marker=dict(colors=[colors.get(label, '#667eea') for label in sentiment_counts.index]),
        textinfo='label+percent',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    
    return fig

def create_source_bar_chart(df):
    """Create a modern bar chart for sources"""
    source_counts = df['source'].value_counts().head(10)
    
    fig = go.Figure(data=[go.Bar(
        x=source_counts.values,
        y=source_counts.index,
        orientation='h',
        marker=dict(
            color=source_counts.values,
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.6)', width=2)
        ),
        text=source_counts.values,
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Articles: %{x}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='Top 10 News Sources',
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Number of Articles',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white'
        ),
        yaxis=dict(
            title='',
            color='white'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=150)
    )
    
    return fig

def create_sentiment_by_source(df):
    """Create sentiment comparison by source"""
    sentiment_by_source = df.groupby('source')['sentiment_compound'].mean().sort_values(ascending=False).head(10)
    
    colors = ['#10b981' if x > 0.05 else '#ef4444' if x < -0.05 else '#6b7280' 
              for x in sentiment_by_source.values]
    
    fig = go.Figure(data=[go.Bar(
        x=sentiment_by_source.values,
        y=sentiment_by_source.index,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.6)', width=2)
        ),
        text=[f"{x:.2f}" for x in sentiment_by_source.values],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Avg Sentiment: %{x:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='Average Sentiment by Source',
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Sentiment Score',
            range=[-0.3, 0.3],
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white',
            zeroline=True,
            zerolinecolor='white',
            zerolinewidth=2
        ),
        yaxis=dict(
            title='',
            color='white'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=150)
    )
    
    return fig

def create_scatter_plot(df):
    """Create sentiment vs length scatter plot"""
    fig = px.scatter(
        df.head(200),
        x='word_count',
        y='sentiment_compound',
        color='sentiment_label',
        size='word_count',
        hover_data=['title', 'source'],
        color_discrete_map={
            'positive': '#10b981',
            'neutral': '#6b7280',
            'negative': '#ef4444'
        },
        title='Sentiment vs Article Length'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            font=dict(size=20, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Word Count',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white'
        ),
        yaxis=dict(
            title='Sentiment Score',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='white',
            zeroline=True,
            zerolinecolor='white'
        ),
        legend=dict(
            title='Sentiment',
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.3)'
        ),
        height=400
    )
    
    return fig

def get_sentiment_emoji(sentiment):
    """Return emoji for sentiment"""
    emoji_map = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }
    return emoji_map.get(sentiment, '😐')

def get_sentiment_color(score):
    """Return color based on sentiment score"""
    if score > 0.05:
        return '#10b981'
    elif score < -0.05:
        return '#ef4444'
    else:
        return '#6b7280'

# ============================================
# MAIN APP
# ============================================

# Header
st.markdown("""
    <h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>
        📰 News Bias Analyzer
    </h1>
    <p style='text-align: center; color: white; font-size: 1.2rem; margin-top: 0;'>
        Analyzing Media Bias Using NLP & Machine Learning
    </p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 About This Project")
    st.markdown("""
    This dashboard analyzes news articles for:
    
    - 📊 **Political Bias Detection**
    - 💭 **Sentiment Analysis**
    - 🔍 **Source Comparison**
    - 📈 **Topic Trends**
    
    ---
    
    ### 🛠️ Tech Stack
    - Python, Transformers
    - NLTK, spaCy
    - Streamlit, Plotly
    - Machine Learning
    
    ---
    
    ### 📁 Dataset Info
    """)

# Load data
df = load_data()

if df is None:
    st.error("❌ Data file not found! Please run the data collection and analysis scripts first.")
    st.info("""
    **Run these commands:**
    1. `python src/newsapi_collector.py`
    2. `python src/preprocessing.py`
    3. `python src/sentiment_analysis.py`
    """)
    st.stop()

# Sidebar dataset info
with st.sidebar:
    st.metric("📰 Total Articles", f"{len(df):,}")
    st.metric("📡 News Sources", f"{df['source'].nunique()}")
    
    avg_sentiment = df['sentiment_compound'].mean()
    sentiment_delta = "Positive" if avg_sentiment > 0 else "Negative"
    st.metric("💭 Avg Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_delta)
    
    st.markdown("---")
    st.markdown("### 🚀 Created By")
    st.markdown("**Gunesh Yeole**")
    st.markdown("[GitHub](https://github.com/GuneshYeole07) | [LinkedIn](https://www.linkedin.com/in/gunesh-yeole-462b8b37a?utm_source=share_via&utm_content=profile&utm_medium=member_ios)")

# Main Dashboard Metrics
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    positive_pct = (df['sentiment_label'] == 'positive').sum() / len(df) * 100
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>😊 Positive</h3>
            <h1 style='color: white; margin: 10px 0;'>{positive_pct:.1f}%</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>{(df['sentiment_label'] == 'positive').sum()} articles</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    neutral_pct = (df['sentiment_label'] == 'neutral').sum() / len(df) * 100
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>😐 Neutral</h3>
            <h1 style='color: white; margin: 10px 0;'>{neutral_pct:.1f}%</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>{(df['sentiment_label'] == 'neutral').sum()} articles</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    negative_pct = (df['sentiment_label'] == 'negative').sum() / len(df) * 100
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>😞 Negative</h3>
            <h1 style='color: white; margin: 10px 0;'>{negative_pct:.1f}%</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>{(df['sentiment_label'] == 'negative').sum()} articles</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    avg_length = df['word_count'].mean()
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>📝 Avg Length</h3>
            <h1 style='color: white; margin: 10px 0;'>{avg_length:.0f}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>words per article</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Explore Articles", "📈 Deep Analysis", "☁️ Word Cloud"])

with tab1:
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiment Distribution")
        fig_donut = create_sentiment_donut(df)
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with col2:
        st.markdown("### Top News Sources")
        fig_sources = create_source_bar_chart(df)
        st.plotly_chart(fig_sources, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Timeline if date available
    if 'published_at' in df.columns or 'date_collected' in df.columns:
        st.markdown("### 📅 Article Timeline")
        date_col = 'published_at' if 'published_at' in df.columns else 'date_collected'
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        timeline = df.groupby(df['date'].dt.date).size().reset_index()
        timeline.columns = ['Date', 'Articles']
        
        fig_timeline = px.area(
            timeline,
            x='Date',
            y='Articles',
            title='Articles Published Over Time',
            color_discrete_sequence=['#667eea']
        )
        
        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(font=dict(color='white'), x=0.5, xanchor='center'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white'),
            height=300
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)

with tab2:
    st.markdown("## 🔍 Explore Articles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sources = ['All'] + sorted(df['source'].unique().tolist())
        selected_source = st.selectbox("📡 Filter by Source", sources)
    
    with col2:
        sentiments = ['All', 'positive', 'neutral', 'negative']
        selected_sentiment = st.selectbox("💭 Filter by Sentiment", sentiments)
    
    with col3:
        sort_options = ['Most Recent', 'Most Positive', 'Most Negative']
        sort_by = st.selectbox("🔄 Sort By", sort_options)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]
    
    # Apply sorting
    if sort_by == 'Most Positive':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=False)
    elif sort_by == 'Most Negative':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=True)
    
    st.markdown(f"### Showing {len(filtered_df)} articles")
    
    # Display articles in a modern card layout
    for idx, row in filtered_df.head(15).iterrows():
        sentiment_color = get_sentiment_color(row['sentiment_compound'])
        sentiment_emoji = get_sentiment_emoji(row['sentiment_label'])
        
        with st.expander(f"{sentiment_emoji} {row['title']}", expanded=False):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**📡 Source:** {row['source']}")
            
            with col2:
                st.markdown(f"**💭 Sentiment:** {row['sentiment_label'].title()}")
            
            with col3:
                st.markdown(f"**📊 Score:** {row['sentiment_compound']:.2f}")
            
            with col4:
                st.markdown(f"**📝 Words:** {row['word_count']}")
            
            # Progress bar for sentiment
            sentiment_normalized = (row['sentiment_compound'] + 1) / 2  # Convert -1 to 1 → 0 to 1
            st.progress(sentiment_normalized)
            
            st.markdown("---")
            st.markdown(f"**Preview:** {row['text'][:400]}...")
            
            if 'url' in row and pd.notna(row['url']):
                st.markdown(f"[🔗 Read Full Article]({row['url']})")

with tab3:
    st.markdown("## 📈 Deep Analysis")
    
    st.markdown("### Sentiment by Source Comparison")
    fig_sentiment_source = create_sentiment_by_source(df)
    st.plotly_chart(fig_sentiment_source, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### Sentiment vs Article Length")
    fig_scatter = create_scatter_plot(df)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Statistical Summary")
        summary_stats = df['sentiment_compound'].describe()
        
        for stat, value in summary_stats.items():
            st.markdown(f"**{stat.title()}:** {value:.3f}")
    
    with col2:
        st.markdown("### 🏆 Most Extreme Articles")
        
        st.markdown("**Most Positive:**")
        most_positive = df.nlargest(1, 'sentiment_compound').iloc[0]
        st.info(f"📰 {most_positive['title']}\n\n**Score:** {most_positive['sentiment_compound']:.3f}")
        
        st.markdown("**Most Negative:**")
        most_negative = df.nsmallest(1, 'sentiment_compound').iloc[0]
        st.warning(f"📰 {most_negative['title']}\n\n**Score:** {most_negative['sentiment_compound']:.3f}")

with tab4:
    st.markdown("## ☁️ Word Cloud")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Options")
        sentiment_filter = st.radio(
            "Filter by sentiment:",
            ['All', 'Positive', 'Negative', 'Neutral']
        )
        
        max_words = st.slider("Max words:", 50, 200, 100)
    
    with col2:
        # Filter data for word cloud
        if sentiment_filter != 'All':
            wc_df = df[df['sentiment_label'] == sentiment_filter.lower()]
        else:
            wc_df = df
        
        if 'text_clean' in wc_df.columns:
            text_col = 'text_clean'
        else:
            text_col = 'text'
        
        all_text = ' '.join(wc_df[text_col].dropna().astype(str))
        
        if all_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='#1e1e1e',
                colormap='viridis',
                max_words=max_words,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('none')
            st.pyplot(fig, transparent=True)
        else:
            st.warning("No text data available for word cloud.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h3>🎓 Built with Python, NLP & Machine Learning</h3>
        <p>Data Science Portfolio Project | 2025</p>
        <p>
            <a href='https://github.com/Gunesh Yeole' style='color: white; margin: 0 10px;'>GitHub</a> |
            <a href='https://www.linkedin.com/in/gunesh-yeole-462b8b37a?utm_source=share_via&utm_content=profile&utm_medium=member_ios' style='color: white; margin: 0 10px;'>LinkedIn</a> |
            <a href='mailto:guneshyeole76@gmail.com.com' style='color: white; margin: 0 10px;'>Email</a>
        </p>
    </div>
""", unsafe_allow_html=True)