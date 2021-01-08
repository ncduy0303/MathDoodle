from fastai.vision.all import *
from utils import *
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import os
import time

st.set_page_config("MathDoodle", "🦈")
st.title("MathDoodle 🦈")

download_models()

option_model = st.sidebar.radio('', ['Mathematical Expressions', 'Geometric Shapes'])

option_upload = st.sidebar.radio('', ['Choose a test image', 'Choose your own image', 'Draw your own image'])
if option_upload == 'Choose a test image':
    test_images = os.listdir('images/')
    test_image = st.sidebar.selectbox('Please select a test image:', test_images)
    file_path = 'images/' + test_image
    img = PILImageBW.create(file_path)
    display_img = PILImage.create(file_path)
    st.image(display_img, use_column_width=True)
    if st.button('Predict'): predict(img, option_model)


if option_upload == 'Choose your own image':
    uploaded_file = st.sidebar.file_uploader("Please upload an image", type="jpg")
    if uploaded_file is not None:
        img = PILImageBW.create(uploaded_file)
        display_img = PILImage.create(uploaded_file)
        st.image(display_img, use_column_width=True)
        if st.button('Predict'): predict(img, option_model)


if option_upload == 'Draw your own image':
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
        display_img = canvas_result.image_data
        img = PILImageBW.create(display_img[:, :, 0].astype(np.uint8))
        if st.button('Predict'): predict(img, option_model)