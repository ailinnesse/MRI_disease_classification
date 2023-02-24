import streamlit as st
from tensorflow import keras, image
from tensorflow import expand_dims
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from PIL import Image
import numpy as np


# Set up credentials
creds = service_account.Credentials.from_service_account_file('mri-classification-9235a58a002b.json')
service = build('drive', 'v3', credentials=creds)

# Define the file ID of the file you want to read
file_id = '1nEG_DJ3JALKz1Ylv40z_b22eEpatjM9K'

# Download the file contents
try:
    request = service.files().get_media(fileId=file_id)
    content = request.execute()
    model_d_no_d = keras.models.load_model(content)
except HttpError as error:
    st.write(f'An error occurred: {error}')



# Model predicting Ailzheimer's or Brain Tumor
model_al_bt = keras.models.load_model('al_bt.hdf5')
# Predict Severity of alzheimer's disease
model_al = keras.models.load_model('model_al.hdf5')
# Predict Brain Tumor Type
model_bt = keras.models.load_model('model_bt.hdf5')


st.title('Do you have Alzheimer\'s disease or Brain tumor?')
st.subheader('Upload the image of the brain MRI and see if it has Alzheimer\'s disease and it\'s severity or Brain tumor and it\'s kind or no disease')
your_image = st.file_uploader(label='Upload your image here', type=["png", "jpg", "jpeg"])
if your_image is not None:
    test_img = Image.open(your_image)
    test_img = keras.preprocessing.image.img_to_array(test_img)
    test_img = expand_dims(test_img, -1)
    test_img = image.grayscale_to_rgb(test_img)
    test_img = keras.preprocessing.image.smart_resize(test_img, (240, 240))
if st.button('Submit'):
    st.image(your_image, width=300)
    pred = model_al_bt.predict(test_img) 
    
    if pred[0] < 0.5:
         st.write('Predicted: No disease found in MRI')
    else:
        pred_dis = model_al_bt.predict(test_img)
        if pred[0] < 0.5:
            pred_al = model_al.predict(test_img)
            pred_al = np.argmax(pred_al, axis = 1)[0]
            al_list = ['Mild', 'Moderate', 'Very Mild']
            st.write('Predicted:', al_list[pred_al])
        
        else:
            pred_bt = model_bt.predict(test_img)
            pred_bt = np.argmax(pred_bt, axis = 1)[0]
            bt_list = ['Glioma', 'Meningioma', 'Pituitary']
            st.write('Predicted:', bt_list[pred_bt])

