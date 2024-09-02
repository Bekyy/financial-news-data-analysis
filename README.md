# Financial News and Stock Price Integration Dataset (FNSPID) Analysis

**Overview**
> The Financial News and Stock Price Integration Dataset (FNSPID) is a comprehensive dataset designed to enhance stock market predictions by integrating quantitative stock price data with qualitative news data. This report presents the results of an exploratory data analysis (EDA) on the dataset, encompassing descriptive statistics, text analysis, time series analysis, quantitative financial analysis using TA-Lib and PyNance, and correlation analysis between news and stock movements,.
> This project focuses on the detailed analysis of a large corpus of financial news data to discover correlations between news sentiment and stock market movements. 

## Dataset Description

1. Financial News Dataset
> The Financial News Dataset consists of quantitative data related to financial news articles. The structure of the data is as follows:
- headline: The title of the news article, often highlighting key financial actions such as stocks hitting highs, price target changes, or company earnings.
- url: The direct link to the full news article.
- publisher: The author or creator of the article.
- date: The publication date and time, including timezone information (UTC-4).
- stock: The stock ticker symbol, which is a unique series of letters assigned to a publicly traded company (e.g., AAPL for Apple).

2. Stock Price Dataset
> The Stock Price Dataset consists of qualitative data for seven companies: Apple (AAPL), Nvidia (NVDA), Meta (META), Amazon (AMZN), Google (GOOG), Tesla (TSLA), and Microsoft (MSFT). The structure of the data includes:
- Ticker: The symbol representing a particular stock.
- Open: The price at which a stock starts trading when the market opens.
- High: The highest price at which a stock traded during the day.
- Low: The lowest price at which a stock traded during the day.
- Close: The final price at which the stock traded when the market closed.
- Volume: The number of shares traded during a specific period.

# Analysis Scope
* Descriptive Statistics: To understand the distribution of headline lengths
* Sentiment Analysis: to categorize headlines as positive, negative, or neutral.
* Topic Modeling: to identify common keywords and topics.
* Time Series Analysis: to identify trends and spikes.
* Quantitative Analysis Using PyNance and TA-Lib
- Moving Averages (MA): Simple Moving Average (SMA), Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
* Correlation between News and Stock Movement: to test the correlation between daily news sentiment scores and stock returns.

## Prerequisites
* Python 3.x: Ensure Python is installed on your system.
* Virtual Environment: Recommended for managing project dependencies.
* Required Libraries:
  - pandas: Data manipulation and analysis.
  - numpy: Numerical operations.
  - matplotlib: Data visualization.
  - seaborn: Statistical visualizations.
  - scikit-learn: for NLP ( topic modeling )
  - textblob/ nltk: sentiment analysis and topic modeling
  - TA-Lib: to calculate financial indicators (MA, RSI, MACD)  

**Installation**

1. **Create a virtual environment:**

   On macOS/Linux:
   ```
   python -m venv venv 
   source venv/bin/activate
   ```
   on windows:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**   
   ``` pip install -r requirements.txt```


**Contributing**

Contributions are welcome!

**License**

This project is licensed under the Apache License. See the LICENSE file for more details.