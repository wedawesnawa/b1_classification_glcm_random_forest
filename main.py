
import os
import cv2
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
import zipfile
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit.components.v1 as components
import tempfile
from PIL import Image
import base64
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed
import joblib

st.set_page_config(page_title="Kelompok B1", layout='centered', initial_sidebar_state='expanded', menu_items=None)




# Bagian Tampilan

st.markdown("""<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

def get_image_html(image_path, alt_text="Image", width=None):

    img = Image.open(image_path)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    width_str = f' width="{width}"' if width else ""
    img_html = f'<img src="data:image/png;base64,{img_str}" alt="{alt_text}"{width_str}>'
    
    return img_html

image_html = get_image_html("logo.png", "Kelompok B1", width=248)

st.markdown(f"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<nav class="navbar fixed-top navbar-expand-lg navbar-light bg-light pb-3 " style="padding-top:40px">
    <div class="container">
        {image_html}
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbar-list-2" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbar-list-2">
            <ul class="navbar-nav ml-auto mt-2 mt-lg-0" style="font-size: 18px !; font-weight: 700 !; font-family: poppins !;">
                <li class="nav-item">
                    <a class="nav-link" href="" target="_blank">About Us</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="" target="_blank">FAQ</a>
                </li>
                <li class="nav-item">
                    <button type="button" class="btn btn-primary btn-md py-2 px-3" style="width: 166px">Sign In</button>
                </li>
            </ul>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;  # Ubah nilai ini sesuai kebutuhan
        padding: 1rem 2rem;  # Padding dalam konten
    }
    .st-emotion-cache-1ab9dzl{
        margin: 50px 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def convert_to_png(image_bytes):
  """
  Converts image data in bytes to PNG format.

  Args:
      image_bytes (bytes): The image data in bytes format.

  Returns:
      bytes: The converted PNG image data.
  """
  img = Image.open(io.BytesIO(image_bytes))
  img_rgb = img.convert('RGB')  # Convert to RGB if needed for the model
  buffer = io.BytesIO()
  img_rgb.save(buffer, format="PNG")
  return buffer.getvalue()

# Convert image to Base64
def image_to_base64(img_bytes, output_size=(64, 64)):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize(output_size)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


with st.container(height=800, border=False):
    col1, col2= st.columns([4,4],gap="large")
    with col1:
        st.image("Vector.png")
        st.markdown("""
            <h1 class="font-weight-bolder" style="font-weight: 900; color: #4788E9;">Classification of Facial Expressions</h1>
            <button type="button" class="btn btn-primary btn-lg px-4 py-2">Get Started 100% Free</button>
        """, unsafe_allow_html=True)
    with col2:
        with st.container(border=True, height=374):

            st.markdown("""
                <style>
                .st-emotion-cache-1gulkj5 {  
                    height: 217px !important;
                    flex-flow: column !important;
                    justify-content: center !important;  
                }
                .st-emotion-cache-u8hs99 {
                    margin-right:0px !important;
                    flex-flow: column !important;
                }
                .st-emotion-cache-1fttcpj{
                    align-items: center !important;
                }
                .st-emotion-cache-7ym5gk{
                    margin-top: 12px !important;
                }
                .st-emotion-cache-1jmvea6{
                    background-color: #4788E9 !important;
                    padding-top: 8px !important;
                    padding-bottom: 8px !important;
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                    width: 100% !important;
                    height: 53px !important;
                    border-radius: 8px !important;
                }
                .st-emotion-cache-1jmvea6 p span{
                    font-size: 16px !important;
                    color : #fff !important
                }
                .st-emotion-cache-ue6h4q{
                    margin-bottom: 12px !important;
                }
                .st-emotion-cache-hqt771{
                    box-shadow: 0 0 32px rgba(0, 0, 0, 0.25) !important;
                }
                </style>
                """, unsafe_allow_html=True)
            
            
            def load_model_and_encoder(model_path, encoder_path):
                try:
                    model = joblib.load(model_path)
                    encoder = joblib.load(encoder_path)
                    return model, encoder
                except FileNotFoundError:
                    st.error(f"File not found. Please ensure the files {model_path} and {encoder_path} exist.")
                    return None, None
                except Exception as e:
                    st.error(f"An error occurred while loading the model: {e}")
                    return None, None

            # Load the trained model and label encoder
            pipeline_rf, label_encoder = load_model_and_encoder('glcm_rf_model.pkl', 'label_encoder.pkl')

            # Define function to extract GLCM features from an image
            def extract_greycomatrix_features(image):
                glcm = graycomatrix(image, distances=[1, 3, 4, 5, 7], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                asm = graycoprops(glcm, 'ASM')[0, 0]
                features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
                return features
            
            uploaded_file = st.file_uploader(':blue[Upload an Image]',type= ['png', 'jpg'], accept_multiple_files=True )
        
        # Initialize a session state to store uploaded images
        if 'uploaded_images' not in st.session_state:
            st.session_state.uploaded_images = []
        if 'results' not in st.session_state:
            st.session_state.results = []

        if uploaded_file:
            for file in uploaded_file:
                st.session_state.uploaded_images.append(file)
            # Conditionally display the submit button
            if st.session_state.uploaded_images:
                if st.button("Submit"):
                    if st.session_state.uploaded_images:
                        if uploaded_file is not None:
                            for uploaded_file in uploaded_file:
                                # Read the image file
                            
                                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)


                                # Display the uploaded image
                                # st.image(img, caption='Uploaded Image', use_column_width=True)

                                # Preprocess the image
                                resized_img = cv2.resize(img, (48, 48))

                                # Extract GLCM features
                                glcm_features = extract_greycomatrix_features(resized_img)

                                # Predict the class
                                glcm_features = glcm_features.reshape(1, -1)  # Reshape to match model input
                                prediction = pipeline_rf.predict(glcm_features)
                                predicted_label = label_encoder.inverse_transform(prediction)

                                # Display the prediction
                                # st.write(f"Predicted Label: {predicted_label[0]}")

                                # Convert to PNG
                                file_bytes_png = convert_to_png(file_bytes)

                                 # Append the result to session state
                                st.session_state.results.append({
                                    "image": file_bytes_png,
                                    "label": predicted_label
                                })
     
            else:
                st.write('No images uploaded yet.')  


if st.session_state.results:
    with st.container(height=900, border=False):
        st.title(':blue[Classification]')

        if not uploaded_file.name.lower().endswith('.png'):
            file_bytes = convert_to_png(file_bytes)

        # Read the uploaded logo file
        df = pd.DataFrame({
            "Image": [image_to_base64(result["image"], output_size=(64, 64)) for result in st.session_state.results], 
            "Prediction": [result['label'] for result in st.session_state.results],
        })

        st.dataframe(
            df,
            column_config={
                "Prediction": st.column_config.TextColumn(),
                "Image": st.column_config.ImageColumn(
                    "Preview Image", help="Streamlit app preview screenshots", width="large"
                ),
            },
            hide_index=False,
            use_container_width=True,
        )

st.markdown("""<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>""", unsafe_allow_html=True)

