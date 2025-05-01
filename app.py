import streamlit as st
import numpy as np 
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

#loading the trained model

model=tf.keras.models.load_model('model.h5')
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
#load Scalers and encoder

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('onehotencoder_geography.pkl','rb') as file:
    onehot_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scl=pickle.load(file)

#streamlit app

st.title("Customer Churn Prediction")

# User input

geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member',[0,1])

#prepare the input data

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    "Tenure":[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    "HasCrCard":[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#OneHotEncoder for the geography column

geo_df_input = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehot_encoder.transform(geo_df_input).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

#merging both the dataframes

input_df=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

#scale the data

input_df_scaled=scl.transform(input_df)

#prediction on the data 

prediction=model.predict(input_df_scaled)
prediction_proba=prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")
if prediction_proba>0.5:
    st.write("Customer likely to churn")
else:
    st.write("Customer is not likely to churn")
