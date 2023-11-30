import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("Decision_model.pkl")

# Function to display highlighted prediction results 
def highlight_prediction(prediction):
    if prediction == 1:
        return '<span style="color: green; font-size: 30px;">Non-Bankruptcy</span>'
    else:
        return '<span style="color: red; font-size: 30px;">Bankruptcy</span>'

# Streamlit app
def main():
    st.title('Bankruptcy Prediction App Using Decision Tree')

    # user input
    st.sidebar.header('Enter Input Features')

    industrial_risk = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5)
    management_risk = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5)
    financial_flexibility = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5)
    credibility = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5)
    competitiveness = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5)
    operating_risk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5)

    # Collect user input into a DataFrame
    user_input = pd.DataFrame({
        'industrial_risk': [industrial_risk],
        'management_risk': [management_risk],
        'financial_flexibility': [financial_flexibility],
        'credibility': [credibility],
        'competitiveness': [competitiveness],
        'operating_risk': [operating_risk]
    })

    # Make predictions
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display prediction result
    st.subheader('Prediction Result')
    if prediction[0]== 1:
        st.markdown('<span style="color: green; font-size: 20px;">Non-Bankruptcy</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: red; font-size: 20px;">Bankruptcy</span>', unsafe_allow_html=True)
        
    st.write(f'The predicted class is: {prediction[0]}')
    st.write(f'Prediction Probabilities: {prediction_proba}')

if __name__ == '__main__':
    main()
