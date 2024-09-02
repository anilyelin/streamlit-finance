import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def calculateSMA(df, k):
    logging = True
    results = []
    i=0
    tmp = k
    for i in range(len(df["Close"])-tmp+1):
        if logging:
            print("\tCurrent Value of i: ",i, " Current Value of k is: ",k)
        current = df[i:k]["Close"].sum()
        result = current / tmp
        i+=1
        k+=1
        results.append(np.round(result,2))
    print(results)
    print()
    x_data = df["Date"][tmp-1:len(df["Date"])]
    result_df = pd.DataFrame(list(zip(x_data, results)), columns=["foo","bar"])
    #result_df = pd.DataFrame(list(zip([1,2,3],[4,-1,60])), columns=["a","b"])
    print(result_df)
    return result_df

st.title("Stock Analysis Dashboard")
ticker = st.text_input("Enter stock ticker symbol: ", "AAPL")
time_interval = st.radio("Select time interval", ["1d", "5d", "1mo", "3mo"])
if st.button("Submit"):
    with st.spinner("Fetching Data"):
        stock_data = yf.Ticker(ticker)
        stock_data_history = stock_data.history(period=time_interval)
        st.write(stock_data_history)
        stock_data_history.reset_index(inplace=True)
        st.subheader("{} Chart for {}".format(time_interval, ticker))
        st.line_chart(stock_data_history, x='Date', y='Close', color="#a338eb")

        st.header("Simple Moving Average")
        df = stock_data_history[['Date','Close']].copy()
        calculateSMA(df,4)
        #days = st.number_input("Enter days", min_value=5, step=1)
        st.line_chart(calculateSMA(df, 4).set_index('foo'))

        
        
        #calculateSMA()
        #col1, col2 = st.columns([1, 1])
        #data = np.random.randn(10, 1)
        

        #col1.subheader("A wide column with a chart")
        #col1.line_chart(data)

        #col2.subheader("A narrow column with a chart")
        #col2.line_chart(data)




data = {
        "calories": [420, 380, 390, 370, 360, 350, 340],
        "duration": [50, 40, 45, 41, 42, 45, 32]
    }
a = pd.DataFrame(data)
        
#df = calculateSMA(a,4)
#st.line_chart(df.set_index('a'))