import streamlit as st
from tensorflow import keras, image
from tensorflow import expand_dims
import numpy as np
import gdown

@st.cache_resource()
def load_model_from_google_drive(link_to_file, model_name):
    '''
    Downloads and loads a TensorFlow model from Google Drive.
    Input: 
    - link_to_file (str): Direct download link to the model file on Google Drive.
    - model_name (str): The name to save the downloaded model file as.
    Output: 
    - Returns the loaded TensorFlow model.
    '''
    url = link_to_file
    output = model_name
    # Download the model from Google Drive
    gdown.download(url, output, quiet=False)
    # Load the model
    model = keras.models.load_model(model_name)   
    return model

# Load models
model_d_nod = load_model_from_google_drive(
    link_to_file='https://drive.google.com/uc?id=1IqWXMwVXZqIA1uAdMc1jTKQVpNlForUA', 
    model_name='disease_no_disease_aug.hdf5'
)
model_al_bt = load_model_from_google_drive(
    link_to_file='https://drive.google.com/uc?id=1W9tH6OEfiWa3_1_DKmHfnYsic2Nem3Uw', 
    model_name='al_bt.hdf5'
)
model_al = load_model_from_google_drive(
    link_to_file='https://drive.google.com/uc?id=1g8dimIN0zT_76U_WuNkmCm12QKz0g91T', 
    model_name='model_al_aug.hdf5'
)
model_bt = load_model_from_google_drive(
    link_to_file='https://drive.google.com/uc?id=1pqFz9IZ7FXV_P1dErQQtrX1CMnZYjF66', 
    model_name='model_bt_aug.hdf5'
)

# Set image size
image_size = 240
st.title('Do you have Alzheimer\'s disease or Brain tumor?')
st.subheader('Please upload a brain MRI image to determine whether it indicates the presence of Alzheimer\'s disease and its severity, a brain tumor and its type, or if there is no evidence of disease.')
your_image = st.file_uploader(label='Upload your image here', type=["png", "jpg", "jpeg"])

# Prepare image for TensorFlow models
if your_image is not None:
    test_img = keras.utils.load_img(
        your_image, target_size=(image_size, image_size), color_mode='grayscale'
    )
    test_img = expand_dims(test_img, -1)
    test_img = image.grayscale_to_rgb(test_img)
    test_img = keras.preprocessing.image.img_to_array(test_img)
    test_img = expand_dims(test_img, 0)

if st.button('Submit'):
    st.image(your_image, width=300)
    # Predict Disease (1) or No Disease (0)
    pred_dis_proba = model_d_nod.predict(test_img, verbose=False)
    pred_dis = (pred_dis_proba > 0.5).astype("int32")
    if pred_dis[0][0] == 0:
        pred_dis_proba_print = round(((1 - pred_dis_proba[0][0]) * 100), 2)
        st.write(f'The MRI image has an {pred_dis_proba_print}% probability of indicating the absence of a disease')
    else:
        # Predict Alzheimer's (0) or Brain tumor (1)
        pred_al_bt = (model_al_bt.predict(test_img, verbose=False) > 0.5).astype("int32")
        if pred_al_bt[0][0] == 0:
            # Predict severity of Alzheimer's
            pred_al_proba = model_al.predict(test_img, verbose=False)
            pred_al = pred_al_proba.argmax(axis=1)[0]
            al_list = ['Mild', 'Moderate', 'Very Mild']
            pred_al_proba_print = round((pred_dis_proba[0][0] * 100), 2)
            st.write(f'The MRI image has an {pred_al_proba_print}% probability of indicating the presence of a {al_list[pred_al]} Alzheimer\'s')
        else:
            # Predict type of brain tumor
            pred_bt_proba = model_bt.predict(test_img, verbose=False)
            pred_bt = pred_bt_proba.argmax(axis=1)[0]
            bt_list = ['Glioma', 'Meningioma', 'Pituitary']
            pred_bt_proba_print = round(pred_bt_proba[0][pred_bt] * 100, 2)
            st.write(f'The MRI image has an {pred_bt_proba_print}% probability of indicating the presence of a {bt_list[pred_bt]} tumor')
