import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import streamlit as st
import math
from io import StringIO 

pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

st.write("FALAQ FORECAST FOR 1 MONTH!")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    data = pd.read_csv(uploaded_file)
    if len(data) > 60 :
        choose = st.selectbox('Select item to forecast',data['product_id'].unique())
        dataframe = data[data['product_id'] == choose]
    else :
        st.write("The product dosen`t have data please select other product")
        

    if st.button('FORECAST'):
        for i in range(0,30):
            dataframe["T_" + str(i+1)] = dataframe.total_sales.shift(-i)
            dataframe.fillna(0.0, inplace=True)
        st.write(dataframe)

        y = dataframe[['total_sales']].fillna(method='ffill')
        y = y.values.reshape(-1, 1)
        DT = dataframe.copy()
        DT.set_index('timestamp', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(y)
        y = scaler.transform(y)

        # generate the input and output sequences
        n_lookback = len(dataframe) - 20  # length of input sequences (lookback period)
        n_forecast = 4  # length of output sequences (forecast period)


        X = []
        Y = []

        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(DT[i - n_lookback: i])
            Y.append(y[i: i + n_forecast])

        X = np.array(X).astype(np.float32)
        Y = np.array(Y).astype(np.float32)

        # fit the model
        model = Sequential()
        model.add(LSTM(100,return_sequences=True, input_shape=(n_lookback, X.shape[2])))
        model.add(Dropout(0.25))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.20))
        model.add(LSTM(units=10,return_sequences=False))
        model.add(Dense(units=n_forecast))
        model.compile(loss='mae', optimizer='adam')

        model.compile(loss='mean_squared_error', optimizer='adam')

        early_stop = EarlyStopping(monitor='loss', patience=2)

        model.fit(X, Y, epochs=100, batch_size=32, verbose=1,
                    callbacks=[early_stop])


        X_ = X[- 1:] # last available input sequence
        X_.shape


        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
        SUM = 0
        for i in Y_ :
            SUM = SUM + i
        st.write("Forecast for 1 Months",math.floor(SUM))


        df_past = dataframe[['total_sales','timestamp']].reset_index()
        df_past = df_past.drop(columns=['index'])
        df_past.rename(columns={'timestamp': 'Date', 'total_sales': 'Actual'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

        df_future = pd.DataFrame(columns=[ 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        st.markdown("More details about forecast")
        st.write(df_future)

        results = df_past.append(df_future).set_index('Date')
        st.line_chart(results)

