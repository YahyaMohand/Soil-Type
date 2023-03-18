import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import cross_val_score
import streamlit as st
from io import StringIO 
import requests



warnings.filterwarnings('ignore')
openai.api_key = "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"


st.write("soil moisture percent")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    st.dataframe(data=df)
    features = df[['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']]
    target = df['Output']
    #features = df[['temperature', 'humidity', 'ph', 'rainfall']]
    labels = df['Output']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)

    predicted_values = RF.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)
    print("RF's Accuracy is: ", x)

    print(classification_report(Ytest,predicted_values))
    N = st.number_input('Insert a Netro')
    P = st.number_input('Insert a P')
    K = st.number_input('Insert a K')
    pH = st.number_input('Insert a pH')
    EC = st.number_input('Insert a EC')
    OC = st.number_input('Insert a OC')
    S = st.number_input('Insert a S')
    Zn = st.number_input('Insert a ZN')
    Fe = st.number_input('Insert a Fe')
    Cu = st.number_input('Insert a Cu')
    Mn = st.number_input('Insert a Mn')
    B = st.number_input('Insert a B')

    data = np.array([[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]])
    prediction = RF.predict(data)
    if prediction == 1:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "كيف يمكن حل مشكلة التربة رطبة و ما هي النباتات المناسبة للتربة رطبة"}
        ]
        )

        st.write(completion.choices[0].message)
        st.write('هذه  التربة  رطبه')
    else :
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "كيف يمكن حل مشكلة التربة الغير رطبة و ما هي النباتات المناسبة للتربة الغير رطبة"}
        ]
        )

        st.write(completion.choices[0].message)
        st.write(' هذه  التربة غير رطبه')
        

    


