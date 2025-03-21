import streamlit as st
import numpy as np
import pickle

# Load the trained model and encoders
with open("flight_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Title of the app
st.title("Flight Price Prediction")

# Dropdowns for categorical features
airlines = list(label_encoders["Airline"].classes_)
sources = list(label_encoders["Source"].classes_)
destinations = list(label_encoders["Destination"].classes_)

airline = st.selectbox("Select Airline", airlines)
source = st.selectbox("Select Source", sources)
destination = st.selectbox("Select Destination", destinations)

# Numeric inputs
departure_time = st.text_input("Enter Departure Time (HH:MM)", "10:00")
arrival_time = st.text_input("Enter Arrival Time (HH:MM)", "12:30")
duration = st.number_input("Duration (in hours)", min_value=1.0, max_value=24.0, value=2.5)
stops = st.slider("Number of Stops", 0, 3)

# Predict button
if st.button("Predict Price"):
    # Encode user inputs
    airline_encoded = label_encoders["Airline"].transform([airline])[0]
    source_encoded = label_encoders["Source"].transform([source])[0]
    destination_encoded = label_encoders["Destination"].transform([destination])[0]
    departure_hour, departure_minute = map(int, departure_time.split(":"))
    arrival_hour, arrival_minute = map(int, arrival_time.split(":"))

    # Prepare input for prediction
    input_data = np.array([[
        airline_encoded, source_encoded, destination_encoded, 
        departure_hour, departure_minute, arrival_hour, arrival_minute, 
        duration, stops
    ]])

    # Predict the price
    price = model.predict(input_data)

    # Display the result
    st.success(f"Estimated Flight Price: ₹{price[0]:,.2f}")
