import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculateSMA(df, k):
    """
    Calculates the Simple Moving Average (SMA) for a given dataframe

    Parameters:
        df      : Dataframe with the financial data for a given stock asset
        k (int) : Represents the number of days for the calculation of SMA

    Returns:
        result_df : Returns a df with the calculated SMA with a given k 
                    which will be used for plotting a graph
    """
    logging.basicConfig(filename='app.log')
    results = []
    i=0
    tmp = k
    for i in range(len(df["Close"])-tmp+1):
        if logging:
            logger.info("Current value of i: %d, Current value of k is: %d", i, k)
        current = df[i:k]["Close"].sum()
        result = current / tmp
        i+=1
        k+=1
        results.append(np.round(result,2))
    logger.info("Results: %s ", results)
    x_data = df["Date"][tmp-1:len(df["Date"])]
    result_df = pd.DataFrame(list(zip(x_data, results)), columns=["foo","bar"])
    print(result_df)
    return result_df

def calculateEMA(df, alpha):
    """"
    Calculates the Exponential Moving Average (EMA) for a given dataframe

    Parameters:
        df           : Dataframe with the financial data for a given stock asset
        alpha (float): Represents the smoothing factor, it is a number between 0 and    

    Returns: 
        results_df   : Returns a df with the calculated EMA for a given alpha
    """
    df = df["Close"]
    shape = df.values.shape
    values = df.values.reshape(shape[::-1])
    result_ema = np.zeros(shape[::-1])
    for i in range(values.shape[0]):
        if i == 0:
            result_ema[0] = values[0]
        else:
            result_ema[i] = alpha * values[i] + (1 - alpha) * result_ema[i-1]
    print(result_ema)
    result_df = pd.DataFrame(result_ema.reshape(shape))
    return result_df
    

st.title("Stock Analysis Dashboard")
ticker = st.text_input("Enter stock ticker symbol: ", "AAPL")
time_interval_col, sma_selection = st.columns([1,1])
time_interval_col.subheader("Timeinterval Selector")
time_interval_value = time_interval_col.radio("Select time interval", ["1d", "5d", "1mo", "3mo"])
sma_selection.subheader("SMA Day Selector")
sma_selection_value = sma_selection.number_input("Enter SMA day parameter", min_value=1, max_value=30)
ema_parameter = st.number_input("Enter EMA parameter", min_value=0.1, max_value=0.9, step=0.1)

if st.button("Submit"):
    with st.spinner("Fetching Data"):
        stock_data = yf.Ticker(ticker)
        stock_data_history = stock_data.history(period=time_interval_value)
        st.write(stock_data_history)
        stock_data_history.reset_index(inplace=True)
        st.subheader("{} Chart for {}".format(time_interval_value, ticker))
        st.line_chart(stock_data_history, x='Date', y='Close', color="#a338eb")

        st.header("Simple Moving Average for {}".format(sma_selection_value))
        df = stock_data_history[['Date','Close']].copy()
        calculateSMA(df,4)
        st.line_chart(calculateSMA(df, sma_selection_value).set_index('foo'))
        st.header("Exponential Moving Average for {}".format(ema_parameter))
        st.line_chart(calculateEMA(df, ema_parameter))

