import streamlit as st
from tensorflow import keras, image
from tensorflow import expand_dims
import numpy as np
import gdown


# 

def load_model(link_to_file, model_name):
    '''
    inspired by https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    Input: 
    link_to_file, str - link to the file in google drive (open the file in new window and copy the id - everything after the last slash, put it after https://drive.google.com/uc?id=)
    model_name, str - name of the model for saving on the disc with .hdf5 at the end
    Output: save models to the disc
    Retrun: tensorflow model
    '''
    url = link_to_file
    output = model_name
    gdown.download(url, output, quiet=False)
    model = keras.models.load_model(model_name)   
    return model
# Model predicting Disease or no Disease
model_d_nod = load_model(link_to_file = 'https://drive.google.com/uc?id=1zA6ONHPEr3u1l-z-ig027mNFw-hO2IiO', model_name = 'disease_no_disease.hdf5')    
# Model predicting Ailzheimer's or Brain Tumor
model_al_bt = load_model(link_to_file = 'https://drive.google.com/uc?id=1nEG_DJ3JALKz1Ylv40z_b22eEpatjM9K', model_name = 'al_bt.hdf5') 
# Predict Severity of alzheimer's disease
model_al = load_model(link_to_file = 'https://drive.google.com/uc?id=1nWUbFrMqA1Cd_M6YdFnTJhfcCk0npbCm', model_name = 'model_al.hdf5')
# Predict Brain Tumor Type
model_bt = load_model(link_to_file = 'https://drive.google.com/uc?id=1FLYVsQRSZPZdtzZn4fExGWR9wk0IzlmK', model_name = 'model_bt.hdf5')

image_size = 240
st.title('Do you have Alzheimer\'s disease or Brain tumor?')
st.subheader('Upload the brain MRI image and see if it has Alzheimer\'s disease and it\'s severity or Brain tumor and it\'s kind or no disease')
your_image = st.file_uploader(label='Upload your image here', type=["png", "jpg", "jpeg"])
if your_image is not None:
    test_img = keras.utils.load_img(your_image, target_size=(image_size, image_size), color_mode='grayscale')
    test_img = expand_dims(test_img, -1)
    test_img = image.grayscale_to_rgb(test_img)
    test_img = keras.preprocessing.image.img_to_array(test_img)
    test_img = expand_dims(test_img, 0)
if st.button('Submit'):
    st.image(your_image, width=300)
    pred_dis = (model_d_nod.predict(test_img, verbose = False) > 0.5).astype("int32")
    if pred_dis[0][0] == 0:
         st.write('You have: no disease')
    else:
        pred_al_bt = (model_al_bt.predict(test_img, verbose = False) > 0.5).astype("int32")
        if pred_al_bt[0][0] == 0:
            pred_al = model_al.predict(test_img, verbose = False).argmax(axis=1)[0]
            al_list = ['Mild', 'Moderate', 'Very Mild']
            st.write('You have:', al_list[pred_al], 'Alzheimer\'s')
        
        else:
            pred_bt = model_bt.predict(test_img, verbose = False).argmax(axis=1)[0]
            bt_list = ['Glioma', 'Meningioma', 'Pituitary']
            st.write('You have:', bt_list[pred_bt], 'tumor')

