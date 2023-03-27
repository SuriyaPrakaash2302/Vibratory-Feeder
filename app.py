import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict, classi

st.title('Velocity Predictor')
st.caption('An advanced ML tool can predict the velocity of a part in a vibratory feeder using part parameters like height, weight, height-diameter ratio, and feeder parameters like voltage and frequency. The tool would utilize supervised learning techniques and a large historical dataset to accurately model the complex relationships between input variables and output velocities.')
st.header("Part parameters")
weight = st.slider('Mass (in grams)', min_value=40, max_value=160, step=10, label_visibility="visible")
height = st.slider('Height (in centimeters)', min_value=30, max_value=200, step=10, label_visibility="visible")
h_hd = st.slider('H_HD', min_value=0.0, max_value=2.0, step=0.1, label_visibility="visible")

st.header("Feeder parameters")
voltage = st.slider('Voltage (Volts)', min_value=0, max_value=259, step=1, label_visibility="visible")
frequency = st.slider('Frequency (Hertz)', min_value=0, max_value=500, step=1, label_visibility="visible")

if st.button("Predict"):
	if (voltage < 100 or frequency<60 or frequency>140 or h_hd<0.2 ):
		st.subheader(" There is no movement for the given parameters")
	else:
		result = classi(np.array([[voltage, frequency, height, h_hd,weight]]))
		if (result == "No"):
			st.subheader(" There is no movement for the given parameters")
		else:
			st.subheader("Yes, there is movement for the given parameters.")
			
			velocity= predict(np.array([[voltage, frequency, height, h_hd,weight]]))
			x= round(velocity[0][0],2)
			st.subheader("The velocity of the part is "+"  " +str(x))
