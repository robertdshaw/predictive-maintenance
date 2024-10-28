import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import streamlit as st
import numpy as np
import pandas as pd
import requests  # To fetch data from an API
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

model = load_model('rul_model.h5')

scaler = MinMaxScaler()

def get_real_time_data():
    response = requests.get("create an API or from kaggle")
    data = response.json()
    df = pd.DataFrame(data)
    
    return df

# Function to preprocess and predict RUL for a machine in real time
def predict_rul_real_time(machine_id, telemetry_data):
    telemetry_data_scaled = scaler.transform(telemetry_data)
    
    predicted_rul = model.predict(telemetry_data_scaled)
    
    return np.expm1(predicted_rul).flatten()[0]

# Streamlit App
st.title("Real-Time RUL Prediction for Machines")

if st.button("Start Real-Time Prediction"):
    st.write("Fetching real-time data...")

    while True:
        real_time_data = get_real_time_data()

        machine_id = st.selectbox("Select Machine", real_time_data['machineID'].unique())
        telemetry_data = real_time_data[real_time_data['machineID'] == machine_id].iloc[:, :-1]  # Drop RUL column if present
        
        predicted_rul = predict_rul_real_time(machine_id, telemetry_data)
        st.write(f"Predicted RUL for Machine {machine_id}: {predicted_rul:.2f}")

        time.sleep(10)  # Fetch new data every 10 seconds
