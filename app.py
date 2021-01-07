from fastai.vision.all import *
from utils import *
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import os
import time

st.title("MathDoodle")

download_url('https://math-doodle-models.s3-ap-southeast-1.amazonaws.com/number.pkl', './number.pkl')
download_url('https://math-doodle-models.s3-ap-southeast-1.amazonaws.com/geometry.pkl', './geometry.pkl')

option_model = st.sidebar.radio('', ['Mathematical Expressions', 'Geometric Shapes'])
if option_model == 'Mathematical Expressions':
    learn = load_learner('number.pkl')
else:
    learn = load_learner('geometry.pkl')

def predict(img, display_img):
    st.image(display_img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)
    if option_model == 'Mathematical Expressions':
        pred = learn.multimodel_predict(img)
        st.success(f"This is the answer to your expression: {pred}")
    else:
        pred, _, probs = learn.predict(img)
        prob = round(torch.max(probs).item() * 100, 2)    
        st.success(f"This is a {pred} with the proability of {prob}%.")

option_upload = st.sidebar.radio('', ['Choose a test image', 'Choose your own image', 'Draw your own image'])

if option_upload == 'Choose a test image':
    test_images = os.listdir('images/')
    test_image = st.sidebar.selectbox('Please select a test image:', test_images)
    file_path = 'images/' + test_image
    img = PILImageBW.create(file_path)
    display_img = PILImage.create(file_path)
    predict(img, display_img)

if option_upload == 'Choose your own image':
    uploaded_file = st.sidebar.file_uploader("Please upload an image", type="jpg")
    if uploaded_file is not None:
        img = PILImageBW.create(uploaded_file)
        display_img = PILImage.create(uploaded_file)
        predict(img, display_img)

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
        if st.button('Predict'): predict(img, display_img)