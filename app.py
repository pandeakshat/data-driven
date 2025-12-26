import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="The Alchemist | Personal OS",
    page_icon="⚡",
    layout="wide"
)

# --- HEADER ---
st.title("⚡ The Alchemist")
st.markdown("*Transforming raw logs into personal insights.*")
st.markdown("---")

# --- 1. INGEST LAYER (Data Connection) ---
st.sidebar.header("1. Data Connection")
source_type = st.sidebar.radio("Select Source:", ["📂 Upload CSV", "🔗 Google Sheet Link"])

df = None

# OPTION A: UPLOAD FILE
if source_type == "📂 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your 'Collect' CSV", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File Uploaded Successfully")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

# OPTION B: GOOGLE SHEET LINK
elif source_type == "🔗 Google Sheet Link":
    sheet_url = st.sidebar.text_input("Paste Published CSV Link:", help="File > Share > Publish to Web > CSV")
    if sheet_url:
        try:
            df = pd.read_csv(sheet_url)
            st.sidebar.success("Connected to Cloud")
        except Exception as e:
            st.sidebar.error("Could not load link. Check permissions.")

# STOP APP IF NO DATA
if df is None:
    st.info("👈 Please connect your data source in the Sidebar to begin.")
    st.stop()

# --- 2. ENGINEER LAYER (Processing) ---
try:
    # 1. Clean Dates: Attempt to handle multiple formats automatically
    # First try assuming MM/DD/YYYY (standard US/Excel)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
    
    # If that failed (resulting in NaT), try DD/MM/YYYY
    if df['Date'].isnull().all():
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    
    # Drop rows where Date is still missing
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')

except Exception as e:
    st.error(f"Date Parsing Error: {e}. Please ensure your 'Date' column is clean.")
    st.stop()

# 2. Machine Learning: Sentiment Analysis (The "Alchemist" Touch)
def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0
    # Returns a score between -1.0 (Negative) and 1.0 (Positive)
    return TextBlob(str(text)).sentiment.polarity

if 'Journal' in df.columns:
    df['Sentiment_Score'] = df['Journal'].apply(get_sentiment)
    # Create a smoothed trend line (7-day rolling average)
    df['Sentiment_Trend'] = df['Sentiment_Score'].rolling(window=7, min_periods=1).mean()
else:
    # Fallback if no journal exists
    df['Sentiment_Score'] = 0
    df['Sentiment_Trend'] = 0

# --- 3. MODEL & DEPLOY LAYER (The Dashboard) ---

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.header("2. Configuration")
days_to_show = st.sidebar.slider("Lookback Window (Days)", 7, 90, 30)

# Filter Data
filtered_df = df.tail(days_to_show)

# KPI SCOREBOARD
col1, col2, col3, col4 = st.columns(4)

# Calculate Metrics
avg_energy = filtered_df['Energy'].mean() if 'Energy' in filtered_df.columns else 0
total_deep = filtered_df['Deep_Work'].sum() if 'Deep_Work' in filtered_df.columns else 0
consistency = filtered_df['Tasks_Done'].std() if 'Tasks_Done' in filtered_df.columns else 0

col1.metric("Avg Energy", f"{avg_energy:.1f}/5", delta=f"{avg_energy - 3:.1f}")
col2.metric("Deep Work Hours", f"{total_deep:.1f}h")
col3.metric("Consistency (Std Dev)", f"±{consistency:.2f}", help="Lower is better. Measures volatility.")
col4.metric("Days Tracked", len(filtered_df))

# VISUALS
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Output vs. Energy")
    if 'Tasks_Done' in filtered_df.columns and 'Energy' in filtered_df.columns:
        chart_data = filtered_df.set_index('Date')[['Tasks_Done', 'Energy']]
        st.line_chart(chart_data)
    else:
        st.warning("Missing 'Tasks_Done' or 'Energy' columns.")

with c2:
    st.subheader("🧠 Mood vs. AI Sentiment")
    
    # 1. Handle Subjective Mood (Map text to numbers if needed)
    if 'Mood' in filtered_df.columns and 'Mood_Score' not in filtered_df.columns:
         mood_map = {'Great': 5, 'Good': 4, 'Neutral': 3, 'Bad': 2, 'Terrible': 1}
         filtered_df['Mood_Score'] = filtered_df['Mood'].map(mood_map)
    
    # 2. Normalize Mood to fit -1 to 1 scale (to match Sentiment Analysis)
    if 'Mood_Score' in filtered_df.columns:
        filtered_df['Mood_Normalized'] = (filtered_df['Mood_Score'] - 3) / 2
        st.line_chart(filtered_df.set_index('Date')[['Sentiment_Trend', 'Mood_Normalized']])
        st.caption("Blue: What you wrote (AI Analysis) | Red: What you felt (Mood Log)")
    else:
        st.info("Ensure you have a 'Mood' column to see this comparison.")

# ADVANCED: THE CORRELATION MATRIX
with st.expander("Show Correlation Heatmap (The Logic Layer)"):
    st.write("What variables actually move together?")
    # Select only numeric columns
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        # Use a gradient to show strong correlations
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
    else:
        st.write("Not enough numeric data.")

# RAW DATA VIEWER
with st.expander("📂 View Raw Database"):
    st.dataframe(filtered_df)