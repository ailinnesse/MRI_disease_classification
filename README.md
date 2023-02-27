# Disease recognition from MRI scans

---
## Problem Statement
Construct an application that helps medical professionals classify the disease from MRI scan. 
This project will focus on Brain MRI scans with classification of severity of Alzeimer's disease and types of Brain tumor.
We will use accuracy to mesure model perfomances and have traning and validation sets that we will use during model fitting and separate testing set to measure the model perfomance on the unseen data.

---

## Table of Contents

1. [EDA](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/01_EDA.ipynb) : Printing the samples of the images and looking at the classes distributions.

2. [Alzheimer](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/02_Alzheimer.ipynb) : Fit the models for Alzheimer's disease severity classification, including pretrined models EfficientNetV2S, Xception and ResNet50

3. [Brain Tumor](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/03_Brain_tumor.ipynb) : Fit the models for Brain Tumor type classification, including pretrined models EfficientNetV2S, Xception and ResNet50

4. [Augmentation](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/04_Augmentation.ipynb) : Augment the images using ImageDataGenerator and Albumentations and fit the models.

5. [Classifying both disease together](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/05_Classifying_both_together.ipynb) : Run four models. Two Binary: Disease-No Disease and Alzheimer's-Brain Tumor. Two Categorical: Severity of Alzhiemer's and Brain tumor type. 

6. [functions](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/functions.py) : Functions used across severall notebooks. 

7. [Streamlit_MRI_classification](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/Streamlit_MRI_classification.py) : Streamlit app code for classifying single image input.

---
## Data

The data was collected from two kaggle datasets:
     [Alzheimer's disease](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)  and
     [Brain Tumor](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

### Datasets
|Dataset|Description|
|---|---|
|[alzheimer](https://github.com/ailinnesse/MRI_disease_classification/tree/main/data/alzheimer)| Train and Test folders with MRI images splited by severity of the disease - NonDemented, VeryMildDemented, MildDemented, ModerateDemented
|[brain_tumor](https://github.com/ailinnesse/MRI_disease_classification/tree/main/data/brain_tumor)| Train and Test folders with MRI images splited by type of brain tumor - no_tumor, glioma_tumor, meningioma_tumor, pituitary_tumor

---
## EDA

### Alzheimer's disease
The classes for Alzheimer's disease are imbalansed with majority images in Non Alzheimer's class and just a few in Moderate. 
My baseline for this model is 50%
![al_avaliable](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Alzheimer's%20disease%20severity%20level%20classes.jpeg) 

### Brain Tumor
The classes for Brain tumor are balansed between different types, but no tumor has about half of the number of images.
My baseline for this model is 30%
![bt_avaliable](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Brain%20Tumor%20classes.jpeg) 

---

## Model Evaluation
I will use Convolutional Neural Networks (CNN) to classify the images
![CNN](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/CNN.jpeg)


### Separate models
I started by finding the best model for each disease separately.

Alzheimer's disease:
My best CNN model had training accuracy 67%.
I have tried tree different pretrained models EfficientNetV2S (accuracy 63%), Xception (accuracy 59%) and my best pretrained model ResNet50 with accuracy 67% - the same as my best CNN model.
![al_resnet](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Alzheimer_ResNet50.jpg) 

Classes for Alzheimer's disease are imbalanced. I tryed applying class weights on ResNet50, but the accuracy dicreased.

Brain Tumor:
My best CNN model had training accuracy 74%.
I have tried tree different pretrained models EfficientNetV2S my best pretrained model (accuracy 75%), Xception (accuracy 67%) and ResNet50 with accuracy 69% 
![bt_effnet](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Brain_tumor_EfficientNetV2S.jpg) 

### Augmentation
To combat overfitting of my models I used Augmentation.
I used Albumentations

![Albumentations](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/augmentation_pipeline_visualized.jpg) 

For both Alzheimer's and Brain tumore the pest performing augmentations were HorizontalFlip, GaussNoise and Rotate (with limit 10)
For Alzheimer's disease the accuracy of the model staied about the same with and without augmentation.
For Brain tumor the accuracy increaced from 75% to 78%.

### Models for application
![app_architecture](https://github.com/ailinnesse/MRI_disease_classification/blob/main/images/Streamlit_app_architecture.jpg) 

Disease - no disease 

Augmentation
The augmentation that worked best in the Augmentation notebook for both Alzheimer's and Brain tumor data gave only 74% accuracy


Alzheimer's disease

Augmentation
The augmentation that worked best in the Augmentation notebook for both Alzheimer's and Brain tumor data gave only 77% accuracy


---
## Conclusions


---
## Further Study


---
## Software Requirements

For this project, we imported pandas, os, matplotlib, cv2, albumentations, sklearn, numpy, mlxtend, tensorflow for streamlit app: streamlit, h5py, gdown 
