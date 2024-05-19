import streamlit as st
import joblib
import numpy as np


scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')


st.set_page_config(page_title='Car Price Recommendation', page_icon='🚗', layout='centered')
st.title('🚗 Customer Car Price Recommendation App')
st.divider()
st.write("""
    Welcome to the Customer Car Price Recommendation App. This app estimates the price 
    of a car based on your age, salary, and net worth.
""")

st.divider()

st.header('Enter Your Details:')
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Enter Age', min_value=18, max_value=90, value=40, step=1)
with col2:
    salary = st.number_input('Enter Salary ($)', min_value=1000, max_value=99999999)
with col3:
    networth = st.number_input('Enter Net Worth ($)', min_value=0, max_value=99999999)

X = [age, salary, networth]

calculate_button = st.button('Calculate Price')
st.divider()

if calculate_button:
    with st.spinner('Calculating...'):
        try:
            X_2 = np.array(X)
            X_array = scaler.transform([X_2])
            prediction = model.predict(X_array)
            
            st.success(f'The estimated car price is ${prediction[0][0]:,.2f}')
            st.write('This is a fair estimation guideline.')
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info('Please enter your details and press the calculate button.')

st.divider()
st.write("""
    **Note:** This estimation is based on the information provided and the model's 
    training data. For a more accurate assessment, please consult a professional.
""")
st.write("Developed by [Axelson Hoang](https://github.com/axhoang)")


st.markdown("""
    <style>
    .stApp {
        background-color: #2e2e2e;
        color: white;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #2e2e2e;
        text-align: center;
        padding: 10px;
        color: white;
    }
    </style>
    <div class="footer">
        <p>Car Price Recommendation App © 2024 | <a href="https://github.com/axhoang" style="color: #4CAF50;">Axhoang.com</a></p>
    </div>
    """, unsafe_allow_html=True)
