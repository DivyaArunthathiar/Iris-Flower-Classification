import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Importing the pickle File :
load_model= pickle.load(open("iris.pkl", 'rb'))
data = pd.read_csv('Iris.csv')

html_temp = """
<div style = "background-color:skyblue ;padding:13px">
<h1 style = "color:black;text-align:center;"> IRIS FLOWER CLASSIFICATION </h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


setosa= Image.open('setosa.jpg')
versicolor= Image.open('versicolor.jpg')
virginica = Image.open('virginica.jpg')
Sepal_Length = st.slider("Sepal Length : ",4.0,8.0)
Sepal_Width = st.slider("Sepal Length :",2.0,4.5)
Petal_Length = st.slider("Petal Length :",1.0,6.9)
Petal_Width = st.slider("Petal Width :",0.1,2.5)


features = [Sepal_Length,Sepal_Width, Petal_Length, Petal_Width ]

final_variables=pd.DataFrame([features],dtype=float)

if st.button('Classify the Flower '):
    prediction = load_model.predict(final_variables)
    (st.markdown(" IRIS SETOSA ") if prediction == 0 else st.markdown(" IRIS VERSICOLOR ") if prediction == 1 else st.markdown
    (" IRIS VIRGINICA "),
    st.image(setosa) if prediction == 0 else st.image(versicolor) if prediction == 1 else st.image(virginica))



