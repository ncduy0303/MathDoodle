from fastai.vision.all import load_learner, torch, PILImage
from utils import *
import streamlit as st
import os
import time

st.title("PhotoMath")

def predict(img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    learn = load_learner('model.pkl')
    pred = learn.predict(img)
    
    st.success(f"This is the answer to your expression: {pred}")


option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':
    test_images = os.listdir('images/')
    test_image = st.selectbox('Please select a test image:', test_images)
    file_path = 'images/' + test_image
    img = PILImageBW.create(file_path)
    predict(img)

else:
    uploaded_file = st.file_uploader("Please upload an image", type="jpg")
    if uploaded_file is not None:
        img = PILImageBW.create(uploaded_file)
        predict(img)