# Disease recognition from MRI scans

---
## Problem Statement
The objective of this project is to develop an application that can assist medical professionals in accurately classifying diseases from MRI scans, focusing on brain MRI scans specifically. The two target diseases are Alzheimer's disease and brain tumors, and the classification will be based on disease severity and tumor type.

To evaluate the performance of my multiclass classification model for diseases, I will be using accuracy as the evaluation metric. This is because I have found that medical professionals have achieved an accuracy of about 75% on this task, and using accuracy as a metric will allow me to compare my model's performance to that of medical professionals.
The dataset will be split into training and validation sets for model fitting and tuning, respectively, and a separate testing set to evaluate the model's performance on new, unseen data.

---

## Table of Contents

1. [EDA](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/01_EDA.ipynb) : Printing the samples of the images and looking at the classes distributions.

2. [Alzheimer](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/02_Alzheimer.ipynb) : Fit the models for Alzheimer's disease severity classification, including pre-trained models EfficientNetV2S, Xception, and ResNet50 to find the model that works best for the data

3. [Brain Tumor](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/03_Brain_tumor.ipynb) : Fit the models for Brain Tumor type classification, including pre-trained models EfficientNetV2S, Xception, and ResNet50 to find the model that works best for the data

4. [Augmentation](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/04_Augmentation.ipynb) : Augment the images using ImageDataGenerator and Albumentations and fit the models.

5. [Classifying both disease together](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/05_Classifying_both_together.ipynb) : Run four models. Two Binary: Disease-No Disease and Alzheimer's-Brain Tumor. Two Categorical: Severity of Alzheimer's and Brain tumor type. Construct and evaluate application for predicting single imput image

6. [functions](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/functions.py) : Functions used across several notebooks. 

7. [Streamlit_MRI_classification](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/Streamlit_MRI_classification.py) : Streamlit app code for classifying single image input.

---
## Data

The data was collected from two Kaggle datasets:
     [Alzheimer's disease](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)  and
     [Brain Tumor](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### Datasets
|Dataset|Description|
|---|---|
|[alzheimer](https://github.com/ailinnesse/MRI_disease_classification/tree/main/data/alzheimer)| Train and Test folders with MRI images split by severity of the disease - NonDemented, VeryMildDemented, MildDemented, ModerateDemented
|[brain_tumor](https://github.com/ailinnesse/MRI_disease_classification/tree/main/data/brain_tumor)| Train and Test folders with MRI images split by type of brain tumor - no_tumor, glioma_tumor, meningioma_tumor, pituitary_tumor

---
## EDA

### Alzheimer's disease
For Alzheimer's disease classification, the classes are imbalanced, with the majority of images belonging to the Non-Alzheimer's class and only a few in the Moderate class. Here, my baseline accuracy is set at 50%.
![al_avaliable](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Alzheimer's%20disease%20severity%20level%20classes.jpeg) 

### Brain Tumor
The classes for the brain tumor classification task are balanced among different types, although the number of images for no tumor class is about half compared to others. For this task, my baseline accuracy is set at 30%.
![bt_avaliable](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Brain%20Tumor%20classes.jpeg) 

---

## Model Evaluation
Convolutional Neural Networks (CNN) were selected as the model architecture for image classification.
![CNN](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/CNN.jpeg)


### Separate models
To begin with, I conducted a search for the best model for each disease classification task individually.

For Alzheimer's disease, the best performing CNN model achieved a training accuracy of 67%. I also experimented with several pre-trained models, including EfficientNetV2S (with an accuracy of 63%), Xception (with an accuracy of 59%), and ResNet50 (with an accuracy of 67% - same as my best CNN model). 
![al_resnet](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Alzheimer_ResNet50.jpg) 

Due to the imbalanced class distribution in the Alzheimer's disease dataset, I attempted to apply class weights to the ResNet50 model but found that this resulted in decreased accuracy.


In the case of brain tumor classification, the best CNN model achieved a training accuracy of 74%. I also tested several pre-trained models, including EfficientNetV2S (my best-performing pre-trained model with an accuracy of 75%), Xception (with an accuracy of 67%), and ResNet50 (with an accuracy of 69%).
![bt_effnet](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Brain_tumor_EfficientNetV2S.jpg) 

### Augmentation
To address the issue of overfitting in the trained models, I implemented augmentation techniques using the Albumentations library. Some examples of the augmentation techniques used include RandomScale, RandomBrightnessContrast, HorizontalFlip, GaussNoise, and Rotate, with a limit of 10 with different probabilities.

By applying these techniques to the training dataset during the model fitting process, the models were better equipped to generalize and perform well on new, unseen data.

Example of the Albumentations augmentations: 

![Albumentations](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/augmentation_pipeline_visualized.jpg) 

The most effective augmentation techniques for both Alzheimer's disease and brain tumors classification were HorizontalFlip, GaussNoise, and Rotate (with a limit of 10).
When applying these augmentation techniques to the Alzheimer's disease classification model, the accuracy of the model remained consistent with and without augmentation.
However, the brain tumors classification model showed an improvement in accuracy, increasing from 75% to 78% when using these augmentation techniques.

### Models for application
After identifying the best models for each disease classification, I proceeded to construct an application that can classify a single image.

The architecture of the application includes the use of the trained deep learning models to predict the disease classification of an MRI scan image. The models are integrated into the application, and the input image is preprocessed using the same techniques used during model training.

The output of the application includes the disease classification of the input image, including the severity of Alzheimer's disease and the type of brain tumor.
![app_architecture](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Streamlit_app_architecture.jpg) 

In this project, four models were developed to classify diseases from MRI scans, including a disease/no-disease model, Alzheimer's disease severity model, Alzheimer's disease and brain tumor binary classification model, and brain tumor type model.

The disease/no-disease model achieved a 80% test accuracy using ResNet50 architecture and augmentation (with the probability of GaussNoise reduced from 0.5 to 0.25). This model I fited using recall as metric instead of accuracy to catch as much disease cases as possible.

The Alzheimer's disease and brain tumor binary classification model achieved exceptional performance with 99.9% test accuracy, with only one image being misclassified. No augmentation was used for this model.

The Alzheimer's disease severity model achieved a 79% test accuracy, with augmentation techniques that differ from the ones used in the augmentation notebook. The probability of GaussNoise was reduced from 0.5 to 0.25.

The brain tumor type model achieved a test accuracy of 76%, and the augmentation techniques and probabilities used were the same as in the augmentation notebook. However, the model still struggled to differentiate Glioma tumors.

Based on the results of the trained models, I selected the best performing models for each disease classification and used them to develop a Streamlit application capable of classifying a single input image. The user can upload an image, and the app displays the image and predicts the class label and corresponding probability using the pre-trained models.
[My Streamlit app](https://ailinnesse-mri-disease--codestreamlit-mri-classification-om06bb.streamlit.app/)

The application developed in this project was utilized to predict the disease classification of MRI scan images in the test set. The evaluation of the application resulted in an accuracy score of 70%.
![app_cm](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/final_app_cm.jpeg) 

---
## Conclusions
I have developed an application that can classify individual brain MRI images and significantly improved on the baseline accuracy of 37%. The application can be a valuable initial tool for medical professionals to analyze MRI images, particularly in the case of meningioma tumors, which are often miscassified by medical professionals.

---
## Recomendations

- Combine predictions on a series of images from one patient (as it is done by medical professionals).
- Expand the MRI scan classification app to include additional diseases such as blood clots, multiple sclerosis, and brain aneurysms.
- Combine the train and test datasets and split them into separate train, validation, and test sets. This may improve the model's learning ability as it appears that the test set contains some different images, which could be causing the lower test accuracy, particularly for Alzheimer's disease.
- Work closely with medical professionals to ensure that the MRI scan classification model is clinically relevant and safe to use.


---
## Software Requirements

For this project, I imported pandas, os, matplotlib, cv2, albumentations, sklearn, numpy, mlxtend, tensorflow. Extra for streamlit app: streamlit, h5py, gdown 
