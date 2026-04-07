import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import json
import subprocess
import sys
from pathlib import Path

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="BOBO",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GOOGLE FONTS + MATERIAL ICONS
# ============================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
""", unsafe_allow_html=True)

# ============================================
# PREMIUM DARK-MODE CSS
# ============================================
st.markdown("""<style>
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0a0a14 0%, #0f1023 35%, #141830 65%, #0d0d1a 100%);
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.3); border-radius: 3px; }

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 15px rgba(0,212,170,0.15); }
    50%      { box-shadow: 0 0 30px rgba(0,212,170,0.3); }
}

/* ── Hero Header ────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInUp 0.8s ease-out;
}
.hero-header h1 {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4aa, #667eea, #a78bfa, #00d4aa);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
    margin-bottom: 0.3rem;
    text-shadow: none;
}
.hero-tagline {
    color: rgba(255,255,255,0.5);
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.04em;
}
.hero-rule {
    width: 80px;
    height: 3px;
    margin: 1rem auto 0;
    background: linear-gradient(90deg, transparent, #00d4aa, transparent);
    border: none;
    border-radius: 2px;
}

/* ── Metric Cards ───────────────────────── */
.metric-card {
    text-align: center;
    padding: 1.6rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fadeInUp 0.6s ease-out both;
}
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 16px 48px rgba(0,0,0,0.35);
}
.metric-card .metric-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: inline-flex; align-items: center; justify-content: center;
    margin: 0 auto 0.5rem; font-size: 0.95rem;
}
.metric-card .metric-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: rgba(255,255,255,0.5); margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-size: 2.2rem; font-weight: 800; color: white; line-height: 1.1;
}
.metric-card .metric-sub {
    font-size: 0.72rem; color: rgba(255,255,255,0.4); margin-top: 0.3rem;
}
.metric-card.green  { border-top: 3px solid #10b981; }
.metric-card.gray   { border-top: 3px solid #6b7280; }
.metric-card.red    { border-top: 3px solid #ef4444; }
.metric-card.purple { border-top: 3px solid #a78bfa; }
.metric-card.green  .metric-value { color: #34d399; }
.metric-card.gray   .metric-value { color: #9ca3af; }
.metric-card.red    .metric-value { color: #f87171; }
.metric-card.purple .metric-value { color: #c4b5fd; }
.metric-card.green  .metric-icon { background: rgba(16,185,129,0.15); color: #34d399; }
.metric-card.gray   .metric-icon { background: rgba(107,114,128,0.15); color: #9ca3af; }
.metric-card.red    .metric-icon { background: rgba(239,68,68,0.15); color: #f87171; }
.metric-card.purple .metric-icon { background: rgba(167,139,250,0.15); color: #c4b5fd; }

/* ── Section Header ─────────────────────── */
.section-header {
    font-size: 1.4rem; font-weight: 700; color: rgba(255,255,255,0.9);
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
h2, h3 { color: rgba(255,255,255,0.9) !important; font-weight: 700 !important; }

/* ── Sidebar ────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1f 0%, #121230 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #00d4aa !important; font-weight: 700;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Tabs ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.03); border-radius: 12px;
    padding: 4px; border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: rgba(255,255,255,0.5) !important;
    font-weight: 600; font-size: 0.9rem; padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4aa 0%, #00b89c 100%) !important;
    color: #0a0a14 !important; box-shadow: 0 4px 15px rgba(0,212,170,0.25);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Buttons ────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg, #00d4aa 0%, #00b89c 100%);
    color: #0a0a14; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-weight: 700;
    transition: all 0.25s ease;
}
.stButton>button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,212,170,0.35);
}

/* ── Select ─────────────────────────────── */
.stSelectbox [data-baseweb="select"] {
    border-radius: 10px; border-color: rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04);
}
.stSelectbox label { color: rgba(255,255,255,0.7) !important; font-weight: 600; }

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important; border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06); font-weight: 600;
    color: rgba(255,255,255,0.85) !important;
}

.stAlert {
    border-radius: 12px; border-left: 4px solid #00d4aa;
    background: rgba(255,255,255,0.03) !important;
}

/* ── Article Cards ──────────────────────── */
.article-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}
.article-card:hover {
    border-color: rgba(0,212,170,0.25); box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    transform: translateY(-2px);
}
.article-card .article-title {
    font-size: 1rem; font-weight: 700; color: rgba(255,255,255,0.9); margin-bottom: 0.5rem;
}
.article-card .article-meta {
    display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.6rem;
}
.article-card .meta-badge {
    font-size: 0.72rem; padding: 0.2rem 0.6rem; border-radius: 20px;
    font-weight: 600; display: inline-block;
}
.badge-green  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.badge-gray   { background: rgba(107,114,128,0.15); color: #9ca3af; border: 1px solid rgba(107,114,128,0.25); }
.badge-red    { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-source { background: rgba(102,126,234,0.15); color: #93a8f4; border: 1px solid rgba(102,126,234,0.25); }
.article-card .article-preview {
    font-size: 0.85rem; color: rgba(255,255,255,0.45); line-height: 1.55; margin-top: 0.4rem;
}
.article-card .sentiment-bar {
    height: 4px; border-radius: 2px; background: rgba(255,255,255,0.06);
    margin-top: 0.75rem; overflow: hidden;
}
.article-card .sentiment-fill {
    height: 100%; border-radius: 2px; transition: width 0.4s ease;
}
.article-card .read-link {
    display: inline-block; margin-top: 0.6rem; font-size: 0.8rem;
    font-weight: 600; color: #00d4aa; text-decoration: none;
}
.article-card .read-link:hover { text-decoration: underline; }

/* ── Stat Pills ─────────────────────────── */
.stat-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.6rem 1rem; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
    margin: 0.25rem; font-size: 0.85rem; color: rgba(255,255,255,0.7);
}
.stat-pill strong { color: #00d4aa; font-weight: 700; }

/* ── Extreme Cards ──────────────────────── */
.extreme-card {
    padding: 1.2rem 1.4rem; border-radius: 14px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.75rem; animation: glowPulse 3s ease-in-out infinite;
}
.extreme-card.positive { border-left: 4px solid #10b981; }
.extreme-card.negative { border-left: 4px solid #ef4444; }
.extreme-card .extreme-label {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin-bottom: 0.4rem;
}
.extreme-card.positive .extreme-label { color: #34d399; }
.extreme-card.negative .extreme-label { color: #f87171; }
.extreme-card .extreme-title {
    font-size: 0.95rem; font-weight: 600; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem;
}
.extreme-card .extreme-score { font-size: 0.8rem; color: rgba(255,255,255,0.45); }

/* ── Footer ─────────────────────────────── */
.footer {
    text-align: center; padding: 2.5rem 0 1.5rem; color: rgba(255,255,255,0.3); font-size: 0.8rem;
}
.footer a {
    color: #00d4aa; text-decoration: none; font-weight: 600; margin: 0 0.5rem;
    transition: color 0.2s ease;
}
.footer a:hover { color: #34d399; text-decoration: underline; }
.footer .footer-divider {
    width: 60px; height: 2px; background: rgba(255,255,255,0.06);
    margin: 1rem auto; border: none;
}
.footer .footer-brand {
    font-size: 0.95rem; color: rgba(255,255,255,0.5); font-weight: 600;
    margin-bottom: 0.25rem;
}

.stRadio > div { gap: 0.3rem; }
.stRadio label, .stSlider label { color: rgba(255,255,255,0.7) !important; }

.stProgress > div > div > div {
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981) !important; border-radius: 4px;
}
</style>""", unsafe_allow_html=True)

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
# CHART THEME HELPERS
# ============================================
CHART_BG = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='rgba(255,255,255,0.7)', size=12),
)

GRID_COLOR = 'rgba(255,255,255,0.05)'
AXIS_STYLE = dict(showgrid=True, gridcolor=GRID_COLOR, color='rgba(255,255,255,0.55)', zeroline=False)

SENTIMENT_COLORS = {
    'positive': '#10b981',
    'neutral':  '#6b7280',
    'negative': '#ef4444',
}

# ============================================
# CHART FUNCTIONS
# ============================================
def create_sentiment_donut(df):
    sentiment_counts = df['sentiment_label'].value_counts()
    total = sentiment_counts.sum()

    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.65,
        marker=dict(
            colors=[SENTIMENT_COLORS.get(l, '#667eea') for l in sentiment_counts.index],
            line=dict(color='#0f0f1a', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(size=13, color='rgba(255,255,255,0.8)', family='Inter'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px;color:rgba(255,255,255,0.4)'>articles</span>",
        x=0.5, y=0.5, font=dict(size=28, color='white', family='Inter'), showarrow=False
    )

    fig.update_layout(
        **CHART_BG,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5,
            font=dict(size=11, color='rgba(255,255,255,0.6)')
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=370
    )
    return fig


def create_source_bar_chart(df):
    source_counts = df['source'].value_counts().head(10)

    fig = go.Figure(data=[go.Bar(
        x=source_counts.values,
        y=source_counts.index,
        orientation='h',
        marker=dict(
            color=source_counts.values,
            colorscale=[[0, '#0e4a3a'], [0.5, '#00d4aa'], [1, '#a78bfa']],
            line=dict(color='rgba(0,212,170,0.3)', width=1),
        ),
        text=source_counts.values,
        textposition='auto',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Articles: %{x}<extra></extra>'
    )])

    fig.update_layout(
        **CHART_BG,
        title=dict(text='Top News Sources', font=dict(size=16, color='rgba(255,255,255,0.85)'), x=0.5, xanchor='center'),
        xaxis={**AXIS_STYLE, 'title': 'Articles'},
        yaxis=dict(title='', color='rgba(255,255,255,0.7)'),
        height=400,
        margin=dict(l=150, t=50, b=40, r=20),
    )
    return fig


def create_sentiment_by_source(df):
    sentiment_by_source = df.groupby('source')['sentiment_compound'].mean().sort_values(ascending=False).head(10)

    colors = ['#10b981' if x > 0.05 else '#ef4444' if x < -0.05 else '#6b7280'
              for x in sentiment_by_source.values]

    fig = go.Figure(data=[go.Bar(
        x=sentiment_by_source.values,
        y=sentiment_by_source.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{x:.2f}" for x in sentiment_by_source.values],
        textposition='auto',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Avg Sentiment: %{x:.3f}<extra></extra>'
    )])

    fig.update_layout(
        **CHART_BG,
        title=dict(text='Average Sentiment by Source', font=dict(size=16, color='rgba(255,255,255,0.85)'), x=0.5, xanchor='center'),
        xaxis={**AXIS_STYLE, 'title': 'Sentiment Score', 'range': [-0.3, 0.3], 'zeroline': True, 'zerolinecolor': 'rgba(255,255,255,0.15)', 'zerolinewidth': 1},
        yaxis=dict(title='', color='rgba(255,255,255,0.7)'),
        height=400,
        margin=dict(l=150, t=50, b=40, r=20),
    )
    return fig


def create_scatter_plot(df):
    fig = px.scatter(
        df.head(200),
        x='word_count',
        y='sentiment_compound',
        color='sentiment_label',
        size='word_count',
        hover_data=['title', 'source'],
        color_discrete_map=SENTIMENT_COLORS,
        title='Sentiment vs Article Length'
    )

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='rgba(255,255,255,0.15)')))

    fig.update_layout(
        **CHART_BG,
        title=dict(font=dict(size=16, color='rgba(255,255,255,0.85)'), x=0.5, xanchor='center'),
        xaxis={**AXIS_STYLE, 'title': 'Word Count'},
        yaxis={**AXIS_STYLE, 'title': 'Sentiment Score', 'zeroline': True, 'zerolinecolor': 'rgba(255,255,255,0.1)'},
        legend=dict(title='Sentiment', font=dict(color='rgba(255,255,255,0.6)'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=50, b=40, l=60, r=20),
        height=420,
    )
    return fig


def get_sentiment_badge_class(label):
    return {'positive': 'badge-green', 'neutral': 'badge-gray', 'negative': 'badge-red'}.get(label, 'badge-gray')


def get_sentiment_fill_color(score):
    if score > 0.05:
        return '#10b981'
    elif score < -0.05:
        return '#ef4444'
    return '#6b7280'

# ============================================
# HERO HEADER
# ============================================
st.markdown("""
<div class="hero-header">
    <h1>BOBO</h1>
    <p class="hero-tagline">Analyzing Media Bias Using NLP &amp; Machine Learning</p>
    <hr class="hero-rule">
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This dashboard analyzes news articles for:

    - **Political Bias Detection**
    - **Sentiment Analysis**
    - **Source Comparison**
    - **Topic Trends**

    ---

    ### Tech Stack
    Python · Transformers · NLTK · spaCy
    Streamlit · Plotly · Machine Learning

    ---

    ### Dataset
    """)

    # ── Refresh Section ────────────────────
    st.markdown("---")
    st.markdown("### Data Refresh")

    refresh_json = Path("data/last_refresh.json")
    is_stale = False
    if refresh_json.exists():
        meta = json.loads(refresh_json.read_text())
        last_refresh_str = meta.get('last_refresh', '')
        try:
            last_dt = datetime.strptime(last_refresh_str, "%Y-%m-%d %H:%M:%S")
            age = datetime.now() - last_dt
            hours_ago = age.total_seconds() / 3600
            if hours_ago < 1:
                age_text = f"{int(age.total_seconds() / 60)} minutes ago"
            elif hours_ago < 24:
                age_text = f"{int(hours_ago)} hours ago"
            else:
                age_text = f"{int(hours_ago / 24)} days ago"
            st.markdown(f"**Last refresh:** {age_text}")
            is_stale = hours_ago >= 24
            if is_stale:
                st.warning("Data is over 24h old - auto-refreshing...")
        except ValueError:
            st.markdown(f"**Last refresh:** {last_refresh_str}")
        st.markdown(f"Total: **{meta.get('total_articles', '?')}** articles")
    else:
        st.caption("No refresh history yet.")
        is_stale = True

    def _run_refresh():
        """Execute the daily_refresh pipeline."""
        result = subprocess.run(
            [sys.executable, "daily_refresh.py"],
            capture_output=True, text=True, encoding='utf-8',
            cwd=str(Path(__file__).resolve().parent)
        )
        return result

    # Auto-refresh if data is stale (>24h)
    if is_stale:
        with st.spinner("Auto-refreshing data - fetching new articles..."):
            result = _run_refresh()
            if result.returncode == 0:
                st.success("Data auto-refreshed successfully.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Auto-refresh failed")
                st.code(result.stderr[-500:] if result.stderr else result.stdout[-500:])

    if st.button("Refresh Now", use_container_width=True):
        with st.spinner("Fetching new articles... this may take a minute."):
            result = _run_refresh()
            if result.returncode == 0:
                st.success("Data refreshed successfully.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Refresh failed")
                st.code(result.stderr[-500:] if result.stderr else result.stdout[-500:])

# Load data
df = load_data()

if df is None:
    st.error("Data file not found. Run the refresh pipeline first.")
    st.info("""
    **Run this command:**
    ```
    python daily_refresh.py
    ```
    Or click the **Refresh Now** button in the sidebar.
    """)
    st.stop()

# Sidebar dataset info
with st.sidebar:
    st.metric("Total Articles", f"{len(df):,}")
    st.metric("News Sources", f"{df['source'].nunique()}")

    avg_sentiment = df['sentiment_compound'].mean()
    sentiment_delta = "Positive" if avg_sentiment > 0 else "Negative"
    st.metric("Avg Sentiment", f"{avg_sentiment:.3f}", delta=sentiment_delta)

    st.markdown("---")
    st.markdown("### Created By")
    st.markdown("**Gunesh Yeole**")
    st.markdown("[GitHub](https://github.com/GuneshYeole07) · [LinkedIn](https://www.linkedin.com/in/gunesh-yeole-462b8b37a)")

# ============================================
# KEY METRICS
# ============================================
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

positive_count = (df['sentiment_label'] == 'positive').sum()
neutral_count  = (df['sentiment_label'] == 'neutral').sum()
negative_count = (df['sentiment_label'] == 'negative').sum()
total = len(df)

with col1:
    pct = positive_count / total * 100
    st.markdown(f"""
    <div class="metric-card green" style="animation-delay: 0.1s;">
        <div class="metric-icon">+</div>
        <div class="metric-label">Positive</div>
        <div class="metric-value">{pct:.1f}%</div>
        <div class="metric-sub">{positive_count} articles</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pct = neutral_count / total * 100
    st.markdown(f"""
    <div class="metric-card gray" style="animation-delay: 0.2s;">
        <div class="metric-icon">=</div>
        <div class="metric-label">Neutral</div>
        <div class="metric-value">{pct:.1f}%</div>
        <div class="metric-sub">{neutral_count} articles</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pct = negative_count / total * 100
    st.markdown(f"""
    <div class="metric-card red" style="animation-delay: 0.3s;">
        <div class="metric-icon">&ndash;</div>
        <div class="metric-label">Negative</div>
        <div class="metric-value">{pct:.1f}%</div>
        <div class="metric-sub">{negative_count} articles</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_len = df['word_count'].mean()
    st.markdown(f"""
    <div class="metric-card purple" style="animation-delay: 0.4s;">
        <div class="metric-icon">W</div>
        <div class="metric-label">Avg Length</div>
        <div class="metric-value">{avg_len:.0f}</div>
        <div class="metric-sub">words per article</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore Articles", "Deep Analysis", "Word Cloud"])

# ── TAB 1: Overview ──────────────────────────
with tab1:
    st.markdown("## Dataset Overview")

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

    # Timeline
    if 'published_at' in df.columns or 'date_collected' in df.columns:
        st.markdown("### Article Timeline")
        date_col = 'published_at' if 'published_at' in df.columns else 'date_collected'
        df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        timeline = df.groupby(df['date'].dt.date).size().reset_index()
        timeline.columns = ['Date', 'Articles']

        fig_timeline = px.area(
            timeline, x='Date', y='Articles',
            title='Articles Published Over Time',
            color_discrete_sequence=['#00d4aa']
        )
        fig_timeline.update_traces(
            fillcolor='rgba(0,212,170,0.12)',
            line=dict(width=2.5, color='#00d4aa')
        )
        fig_timeline.update_layout(
            **CHART_BG,
            title=dict(font=dict(size=16, color='rgba(255,255,255,0.85)'), x=0.5, xanchor='center'),
            xaxis={**AXIS_STYLE, 'title': ''},
            yaxis={**AXIS_STYLE, 'title': 'Articles'},
            margin=dict(t=50, b=40, l=60, r=20),
            height=320,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

# ── TAB 2: Explore Articles ──────────────────
with tab2:
    st.markdown("## Explore Articles")

    col1, col2, col3 = st.columns(3)

    with col1:
        sources = ['All'] + sorted(df['source'].unique().tolist())
        selected_source = st.selectbox("Filter by Source", sources)

    with col2:
        sentiments = ['All', 'positive', 'neutral', 'negative']
        selected_sentiment = st.selectbox("Filter by Sentiment", sentiments)

    with col3:
        sort_options = ['Most Recent', 'Most Positive', 'Most Negative']
        sort_by = st.selectbox("Sort By", sort_options)

    # Apply filters
    filtered_df = df.copy()
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == selected_sentiment]

    if sort_by == 'Most Positive':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=False)
    elif sort_by == 'Most Negative':
        filtered_df = filtered_df.sort_values('sentiment_compound', ascending=True)

    st.markdown(f"**Showing {len(filtered_df)} articles**")
    st.markdown("<br>", unsafe_allow_html=True)

    # Display articles as glassmorphic cards
    for idx, row in filtered_df.head(15).iterrows():
        badge_cls = get_sentiment_badge_class(row['sentiment_label'])
        fill_color = get_sentiment_fill_color(row['sentiment_compound'])
        fill_pct = (row['sentiment_compound'] + 1) / 2 * 100

        preview = str(row['text'])[:350].replace('<', '&lt;').replace('>', '&gt;')
        title_safe = str(row['title']).replace('<', '&lt;').replace('>', '&gt;')
        source_safe = str(row['source']).replace('<', '&lt;').replace('>', '&gt;')

        link_html = ""
        if 'url' in row.index and pd.notna(row.get('url')):
            link_html = f'<a class="read-link" href="{row["url"]}" target="_blank">Read Full Article &rarr;</a>'

        st.markdown(f"""
        <div class="article-card">
            <div class="article-title">{title_safe}</div>
            <div class="article-meta">
                <span class="meta-badge badge-source">{source_safe}</span>
                <span class="meta-badge {badge_cls}">{row['sentiment_label'].title()}</span>
                <span class="meta-badge badge-gray">{row['sentiment_compound']:.2f}</span>
                <span class="meta-badge badge-gray">{row['word_count']} words</span>
            </div>
            <div class="sentiment-bar">
                <div class="sentiment-fill" style="width: {fill_pct:.1f}%; background: {fill_color};"></div>
            </div>
            <div class="article-preview">{preview}...</div>
            {link_html}
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3: Deep Analysis ─────────────────────
with tab3:
    st.markdown("## Deep Analysis")

    st.markdown("### Sentiment by Source Comparison")
    fig_sentiment_source = create_sentiment_by_source(df)
    st.plotly_chart(fig_sentiment_source, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Sentiment vs Article Length")
    fig_scatter = create_scatter_plot(df)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Statistical Summary")
        summary_stats = df['sentiment_compound'].describe()

        pills_html = '<div style="display:flex; flex-wrap:wrap; gap:0.25rem;">'
        for stat, value in summary_stats.items():
            pills_html += f'<div class="stat-pill"><span>{stat.title()}</span> <strong>{value:.3f}</strong></div>'
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

    with col2:
        st.markdown("### Most Extreme Articles")

        most_positive = df.nlargest(1, 'sentiment_compound').iloc[0]
        pos_title = str(most_positive['title']).replace('<', '&lt;').replace('>', '&gt;')
        st.markdown(f"""
        <div class="extreme-card positive">
            <div class="extreme-label">MOST POSITIVE</div>
            <div class="extreme-title">{pos_title}</div>
            <div class="extreme-score">Score: {most_positive['sentiment_compound']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        most_negative = df.nsmallest(1, 'sentiment_compound').iloc[0]
        neg_title = str(most_negative['title']).replace('<', '&lt;').replace('>', '&gt;')
        st.markdown(f"""
        <div class="extreme-card negative">
            <div class="extreme-label">MOST NEGATIVE</div>
            <div class="extreme-title">{neg_title}</div>
            <div class="extreme-score">Score: {most_negative['sentiment_compound']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 4: Word Cloud ────────────────────────
with tab4:
    st.markdown("## Word Cloud")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Options")
        sentiment_filter = st.radio(
            "Filter by sentiment:",
            ['All', 'Positive', 'Negative', 'Neutral']
        )
        max_words = st.slider("Max words:", 50, 200, 100)

    with col2:
        if sentiment_filter != 'All':
            wc_df = df[df['sentiment_label'] == sentiment_filter.lower()]
        else:
            wc_df = df

        text_col = 'text_clean' if 'text_clean' in wc_df.columns else 'text'
        all_text = ' '.join(wc_df[text_col].dropna().astype(str))

        if all_text.strip():
            wordcloud = WordCloud(
                width=900,
                height=450,
                background_color='#0f0f1a',
                colormap='plasma',
                max_words=max_words,
                relative_scaling=0.5,
                min_font_size=10,
                contour_width=0,
            ).generate(all_text)

            fig, ax = plt.subplots(figsize=(13, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('#0f0f1a')
            st.pyplot(fig, transparent=True)
            plt.close(fig)
        else:
            st.warning("No text data available for word cloud.")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <hr class="footer-divider">
    <p class="footer-brand">
        Built with Python, NLP &amp; Machine Learning
    </p>
    <p>Data Science Portfolio Project &middot; 2025</p>
    <p>
        <a href="https://github.com/GuneshYeole07">GitHub</a> &middot;
        <a href="https://www.linkedin.com/in/gunesh-yeole-462b8b37a">LinkedIn</a> &middot;
        <a href="mailto:guneshyeole76@gmail.com">Email</a>
    </p>
</div>
""", unsafe_allow_html=True)