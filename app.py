import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Velocity Predictor')
st.text("Part parameters")
weight = st.slider('Mass', min_value=40, max_value=160, step=10)
height = st.slider('Height', min_value=30, max_value=200, step=10)
h_hd = st.slider('H_HD', min_value=0.0, max_value=2.0, step=0.1)

st.text("Feeder parameters")
voltage = st.slider('voltage', min_value=0, max_value=259, step=1)
frequency = st.slider('frequency', min_value=0, max_value=500, step=1)

if st.button("Predict"):
    result = predict(np.array([[voltage, frequency, height, h_hd,weight]]))
    st.text(result[0])
