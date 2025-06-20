import numpy as np
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

#load the model
model = joblib.load("model_1.pkl")
scale = joblib.load("scaled.pkl")

#streamlit app title

st.title('Machine Learning Model Deployment')
st.write('Enter your Medical details to know about your diabetic status')

#define the input fields
st.sidebar.header("Your Medical records")
preg= st.sidebar.number_input('preg',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
plas= st.sidebar.number_input('plas',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
pres= st.sidebar.number_input('pres',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
skin= st.sidebar.number_input('skin',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
test= st.sidebar.number_input('test',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
mass= st.sidebar.number_input('mass',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
pedi= st.sidebar.number_input('pedi',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)
age= st.sidebar.number_input('age',min_value=0.0, max_value=100.0, value = 0.0, step=0.1)

input_data = np.array([[preg, plas, pres, skin, test, mass, pedi, age]])
scaled_input = scale.transform(input_data)

if st.sidebar.button('Predict'):
    prediction = model.predict(scaled_input)
    if prediction[0] ==0:
        st.success('You are NOT DIABETIC')
    else:
        st.success('You are DIABETIC')
        
 #   st.success(f'Prediction for your given medical data is : {prediction[0]}')



## run the file from terminal -- streamlit run stream.py
## if the above doesnt work try this --  python -m streamlit run 6stream.py
## to rerun at terminal again and to get the prompt - press ctrl C
## create requirements file
## goto github and upload the files - model_1, scaled, stream and requierements
## goto streamlit cloud and connect with github acct
## goto create app - choose the repo and file that contains the streamlit code(stream.py)
