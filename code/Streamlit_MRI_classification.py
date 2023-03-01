import streamlit as st
from tensorflow import keras, image
from tensorflow import expand_dims
import numpy as np
import gdown

@st.cache_resource()
def load_model_from_google_drive(link_to_file, model_name):
    '''
    inspired by https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    Input: 
    link_to_file, str - link to the file in google drive (open the file in a new window and copy the id - everything after the last slash, put it after https://drive.google.com/uc?id=)
    model_name, str - the name of the model for saving on the disc with .hdf5 at the end
    Output: save models to the disc
    Return: TensorFlow model
    '''
    url = link_to_file
    output = model_name
    # download the model from google drive
    gdown.download(url, output, quiet=False)
    # load the model
    model = keras.models.load_model(model_name)   
    return model

# Model predicting Disease or no Disease
model_d_nod = load_model_from_google_drive(link_to_file = 'https://drive.google.com/uc?id=1YmZyLpmM6FKjOgViibGTTHb1Jgz3spli', model_name = 'disease_no_disease_aug.hdf5')    
# Model predicting Ailzheimer's or Brain Tumor
model_al_bt = load_model_from_google_drive(link_to_file = 'https://drive.google.com/uc?id=1nEG_DJ3JALKz1Ylv40z_b22eEpatjM9K', model_name = 'al_bt.hdf5') 
# Predict Severity of alzheimer's disease
model_al = load_model_from_google_drive(link_to_file = 'https://drive.google.com/uc?id=12n6XMrG1LN26x8R_HaHnm-vvAKrATm4b', model_name = 'model_al_aug.hdf5')
# Predict Brain Tumor Type
model_bt = load_model_from_google_drive(link_to_file = 'https://drive.google.com/uc?id=1GE1y3dyLTrnUHrPWdjK117AyF0u-YBdJ', model_name = 'model_bt_aug.hdf5')

# set image size
image_size = 240
st.title('Do you have Alzheimer\'s disease or Brain tumor?')
st.subheader('Please upload a brain MRI image to determine whether it indicates the presence of Alzheimer\'s disease and its severity, a brain tumor and its type, or if there is no evidence of disease.')
your_image = st.file_uploader(label='Upload your image here', type=["png", "jpg", "jpeg"])
# Prepare image for TensorFlow models
if your_image is not None:
    test_img = keras.utils.load_img(your_image, target_size=(image_size, image_size), color_mode='grayscale') # Read grayscale image and set size
    test_img = expand_dims(test_img, -1) # Change from (image_size, image_size) to (image_size, image_size, 1)
    test_img = image.grayscale_to_rgb(test_img) # Change from (image_size, image_size, 1) to (image_size, image_size, 3)
    test_img = keras.preprocessing.image.img_to_array(test_img)
    test_img = expand_dims(test_img, 0) # Change to (1, image_size, image_size, 3) for TensorFlow model 
if st.button('Submit'):
    st.image(your_image, width=300)
    # Predict Disease (1) or No Disease (0)
    pred_dis_proba = model_d_nod.predict(test_img, verbose = False)
    pred_dis = ((pred_dis_proba) > 0.5).astype("int32") 
    if pred_dis[0][0] == 0:
        pred_dis_proba_print = round(((1 - pred_dis_proba[0][0]) * 100), 2) 
        st.write(f'The MRI image has an {pred_dis_proba_print}% probability of indicating the absence of a disease')
    else:
        # Predict Alzheimer's (0) or Brain tumor(1)
        pred_al_bt = (model_al_bt.predict(test_img, verbose = False) > 0.5).astype("int32")
        if pred_al_bt[0][0] == 0:
            # Predict severity of the Alzheimer's
            pred_al_proba = model_al.predict(test_img, verbose = False)
            pred_al = pred_al_proba.argmax(axis=1)[0]
            al_list = ['Mild', 'Moderate', 'Very Mild']
            pred_al_proba_print = round(((pred_dis_proba[0][0]) * 100), 2)
            st.write(f'The MRI image has an {pred_al_proba_print}% probability of indicating the presence of a {al_list[pred_al]} Alzheimer\'s')
        else:
            # Predict type of brain tumor
            pred_bt_proba = model_bt.predict(test_img, verbose = False)
            pred_bt = pred_bt_proba.argmax(axis=1)[0]
            bt_list = ['Glioma', 'Meningioma', 'Pituitary']
            pred_bt_proba_print = round(pred_bt_proba[0][pred_bt] * 100, 2) 
            st.write(f'The MRI image has an {pred_bt_proba_print}% probability of indicating the presence of a {bt_list[pred_bt]} tumor')

