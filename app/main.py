import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr

# Function to load a CSV file into a DataFrame
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def plot_stock_data(df, columns, title='Stock Data'):
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=f'{col}')
        else:
            st.warning(f"Warning: '{col}' not found in DataFrame.")
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_Integrated_data(df1, df2):
    # df1['Date'] = pd.to_datetime(df1.index)
    df1['daily_return'] = df1['Close'].pct_change()

    # Merge sentiment data with stock data
    merged_df = pd.merge(df1, df2, on='Date', how='inner')
    # Drop rows with NaN values (which occur due to pct_change)
    merged_df.dropna(subset=['daily_return', 'sentiment'], inplace=True)

    # Calculate the Pearson correlation coefficient
    sentiment_return_corr, p_value = pearsonr(merged_df['sentiment'], merged_df['daily_return'])
    sentiment_closing_corr, p_value = pearsonr(merged_df['sentiment'], merged_df['Close'])

    # Plot sentiment vs daily return
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(merged_df['Date'], merged_df['sentiment'], label='Sentiment')
    ax1.plot(merged_df['Date'], merged_df['daily_return'], label='Daily Return')
    ax1.set_xlabel('Date')
    ax1.set_title('Sentiment vs Daily Return')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot sentiment vs close price
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(merged_df['Date'], merged_df['sentiment'], label='Sentiment')
    ax2.plot(merged_df['Date'], merged_df['Close'], label='Close Price')
    ax2.set_xlabel('Date')
    ax2.set_title('Sentiment vs Close Price')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    return sentiment_closing_corr, sentiment_return_corr

def prepare_news_data(df):
    timezone_pattern = re.compile(r'[\+\-]\d{2}:\d{2}')
    has_timezone = df['Date'].apply(lambda x: bool(timezone_pattern.search(str(x))) if pd.notnull(x) else False)
    
    df_with_timezone = df[has_timezone]
    df_without_timezone = df[~has_timezone]

    df_with_timezone.reset_index(drop=True, inplace=True)
    df_without_timezone.reset_index(drop=True, inplace=True)

    df_with_timezone['Date'] = pd.to_datetime(df_with_timezone['Date'], errors='coerce')
    df_with_timezone['Date'] = df_with_timezone['Date'].dt.date

    df_without_timezone['Date'] = pd.to_datetime(df_without_timezone['Date'], errors='coerce')
    df_without_timezone['Date'] = df_without_timezone['Date'].dt.date

    concatenated_df = pd.concat([df_with_timezone, df_without_timezone], ignore_index=True)
    concatenated_df.sort_values(by='Date', inplace=True)
    concatenated_df.reset_index(drop=True, inplace=True)
    return concatenated_df

def sentiment_analysis(df):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    # Calculate the sentiment of the headlines
    df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(text=x)['compound'])
    df['sentiment_category'] = pd.cut(df['sentiment'], bins=[-1, -0.5, -0.0001, 0.5, 1], labels=['Very Negative', 'Negative', 'Neutral', 'Positive'])
    return df

# Main function to run the Streamlit dashboard
def financial_news_stock_dashboard():
    st.set_page_config(page_title="Financial News and Stock Price Integration Data Analysis",
                       page_icon=":bar_chart:",
                       layout="wide")
    st.title("Financial News and Stock Price Integration Data Analysis")

    # Sidebar options for the type of plot and columns to plot
    with st.sidebar:
        st.header("Upload and Plot Configuration")
        uploaded_file1 = st.file_uploader("Upload a Stock file", type="csv")
        uploaded_file2 = st.file_uploader("Upload a News file", type="csv")
        
        if uploaded_file1 and uploaded_file2:
            df1 = load_data(uploaded_file1)
            df2 = load_data(uploaded_file2)
            
            if 'Date' in df1.columns and 'Date' in df2.columns:
                df2 = sentiment_analysis(prepare_news_data(df2))
                columns_to_plotStock = st.multiselect("Select Columns to Plot", df1.columns.to_list())
                start_index = st.slider("Select Start Index", 0, len(df1)-1, 0)
                end_index = st.slider("Select End Index", 0, len(df1)-1, len(df1)-1)
                generate_stock_plot = st.button("Generate Stock Plot")
                generate_Integrated_plot = st.button("Generate Integrated Plot")
            else:
                st.error("The uploaded files do not contain the required 'Date' column.")
        else:
            st.info("Please upload both CSV files to get started.", icon="ℹ️")

    if uploaded_file1 and uploaded_file2 and 'Date' in df1.columns and 'Date' in df2.columns:
        df1['Date'] = pd.to_datetime(df1['Date'])
        df=df1.copy()
        df = df.set_index('Date')
        df2['Date'] = pd.to_datetime(df2['Date'])
        if 'columns_to_plotStock' in locals() and generate_stock_plot:
            plot_stock_data(df[start_index:end_index], columns_to_plotStock)
        if generate_Integrated_plot:
            close_corr, ret_corr = plot_Integrated_data(df1, df2)
            st.write(f'Pearson correlation coefficient for sentiment and Close: {close_corr}')
            st.write(f'Pearson correlation coefficient for sentiment and daily return: {ret_corr}')
    else:
        st.info("Upload files and configure the plot using the sidebar.", icon="ℹ️")

if __name__ == "__main__":
    financial_news_stock_dashboard()
