import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error


# Load the dataset
#df = pd.read_excel('Day-wise planets degree and temperature.xlsx')

# Load the trained models
regressor_max = load("regressor1.pkl")
regressor_min = load("regressor2.pkl")

# Asking user to select day, month, and year
day_temp = st.number_input("Enter Day:", min_value=1, max_value=31, value=15)
month_temp = st.number_input("Enter Month:", min_value=1, max_value=12, value=6)
year_temp = st.number_input("Enter Year:", min_value=1900, max_value=2100, value=2023)

# Find the index of the specified date based on day and month
target_month_day = pd.to_datetime(f"{day_temp}-{month_temp}", format='%d-%m')
index_of_date = df[df['Date'].dt.strftime('%d-%m') == target_month_day.strftime('%d-%m')].index

y_max_mean = df['Max Temperature'].mean()
y_min_mean = df['Min Temperature'].mean()

# Display the index and degrees of the selected planet and the rest
if not index_of_date.empty:
    closest_index_of_date = index_of_date[0]
    st.subheader("Selected Planet Degrees:")
    st.write("Index:", closest_index_of_date)



    # Display degrees of selected planets in number_input forms
    selected_planets_degrees = df.iloc[closest_index_of_date, 3:].tolist()
    for i, degree in enumerate(selected_planets_degrees):
        selected_planets_degrees[i] = st.number_input(f"Enter {df.columns[i+3]}:", value=degree, step=1.0)
    
    # Make temperature prediction
    input_data_temp = np.array([[day_temp, month_temp, year_temp]+selected_planets_degrees])
    max_temp_prediction = regressor_max.predict(input_data_temp)[0]
    min_temp_prediction = regressor_min.predict(input_data_temp)[0]

    # Display temperature predictions in red font
    st.subheader("Temperature Predictions:")
    st.markdown(f"Predicted Max Temperature: <span style='color: red;'>{round(max_temp_prediction, 2)} °C</span>", unsafe_allow_html=True)
    st.markdown(f"Predicted Min Temperature: <span style='color: red;'>{round(min_temp_prediction, 2)} °C</span>", unsafe_allow_html=True)
    

    actual_max_temp = df.loc[closest_index_of_date, 'Max Temperature']
    actual_min_temp = df.loc[closest_index_of_date, 'Min Temperature']



    s_max = (mean_squared_error([actual_max_temp], [max_temp_prediction]))**(1/2)
    accuracy_max = round((1 - s_max / (0.1 * y_max_mean)) * 100, 2)


    s_min = (mean_squared_error([actual_min_temp], [min_temp_prediction]))**(1/2)
    accuracy_min = round((1 - s_min / (0.1 * y_min_mean)) * 100, 2)


    st.subheader("Accuracy of Temperature:")
    st.markdown(f"Accuracy for Max Temperature: <span style='color: green;'>{accuracy_max} %</span>",unsafe_allow_html=True)
    st.markdown(f"Accuracy for Min Temperature: <span style='color: green;'>{accuracy_min} %</span>",unsafe_allow_html=True)



else:
    st.subheader("No close match found for the selected degree.")



