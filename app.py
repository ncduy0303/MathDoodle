from fastai.vision.all import load_learner, torch, PILImage
from utils import *
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import os
import time

st.title("MathDoodle")

def predict(img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    learn = load_learner('model.pkl')
    pred = learn.predict(img)
    
    st.success(f"This is the answer to your expression: {pred}")


option = st.radio('', ['Choose a test image', 'Choose your own image', 'Draw your own image'])

if option == 'Choose a test image':
    test_images = os.listdir('images/')
    test_image = st.selectbox('Please select a test image:', test_images)
    file_path = 'images/' + test_image
    img = PILImageBW.create(file_path)
    predict(img)

if option == 'Choose your own image':
    uploaded_file = st.file_uploader("Please upload an image", type="jpg")
    if uploaded_file is not None:
        img = PILImageBW.create(uploaded_file)
        predict(img)

if option == 'Draw your own image':
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="" if bg_image else bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=300,
        width=700,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        data = canvas_result.image_data
        data = data[:, :, 0].astype(np.uint8)
        img = PILImageBW.create(data)
        if st.button('Calculate'): predict(img)