# Disease recognition from MRI scans

---
## Problem Statement
Construct a tool that helps medical professionals classify the disease from MRI scan. 
This project will focus on Brain MRI scans with classification of severity of Alzeimer's disease and types of Brain tumor.
We will use accuracy to mesure model perfomances and have traning and validation sets that we will use during model fitting and separate testing set to measure the model perfomance on the unseen data.

---

## Table of Contents

1. [EDA](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/01_EDA.ipynb) : Printing the samples of the images and looking at the classes distributions.

2. [Alzheimer](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/02_Alzheimer.ipynb) : Fit the models for Alzheimer's disease severity classification, including pretrined models EfficientNetV2S, Xception and ResNet50

3. [Brain Tumor](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/03_Brain_tumor.ipynb) : Fit the models for Brain Tumor type classification, including pretrined models EfficientNetV2S, Xception and ResNet50

4. [Augmentation](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/04_Augmentation.ipynb) : Augment the images using ImageDataGenerator and Albumentations and fit the models.

5. [Classifying both disease together](https://github.com/ailinnesse/MRI_disease_classification/blob/main/code/05_Classifying_both_together.ipynb) : Run four models. Two Binary: Disease-No Disease and Alzheimer's-Brain Tumor. Two Categorical: Severity of Alzhiemer's and Brain tumor type. 
---
## Data

The data was collected from two kaggle datasets:
    * [Alzheimer's disease](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
    * [Brain Tumor](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

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

After testing out different time series models, we picked our two best performing models, recursive multi-step forecasting and SARIMAX and explored how they performed for 3 different time periods, with and without exogenous variables.

I. Forecast Models Before Pandemic

* Forecasting 'counseling' searches for the Most Restricted States benefited from including exogenous features (COVID Restrictions) for both SARIMA and Recursive multi-step models. 
![start_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2020-05-30.jpeg) 

* Forecasting mental health related search terms in Least Restricted states using recursive multi-step forecasting did not benefit from adding exogenous features (COVID Restrictions) and only improved the performance (reduced the MSE) for SARIMAX in forecasting 'depression' searches.
![start_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2020-05-30.jpeg) 

II. Forecast Models During Pandemic
* Both SARIMA and Recursive multi-step models for the Most Restricted States were improved when it came to forecasting 'anxiety', 'mental health' and particularly 'depression', when exogenous features were included in the models.
![middle_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2020-09-30.jpeg)

* The recursive multistep forecasting models for the least restricted states were improved for 'depression', 'anxiety' and particularly 'mental health' searches by including exogenous features whereas the SARIMA with exogenous features only performed better for this group of states in predicting 'addiction' searches.
![middle_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2020-09-30.jpeg)

III. Forecast Models After Pandemic
* Most restricted states did not benefit from adding exogenous features for both SARIMAX and recursive multi-step forecasting models in any of the search terms. However, adding restrictions as exogenous features improved the performance of forecasting:
    * 'depression', 'anxiety' and 'mental health' in recursive multi-step forecasting models
    * 'mental health' searches using the SARIMAX model  
![end_most](https://github.com/MakenaJones/mental_health_searches/blob/main/images/most_mse_diff_2021-01-01.jpeg)

* In the least restricted states,forecasting 'depression' and 'counseling' searches benefited from adding COVID Restrictions as exogenous features in both SARIMAX and recursive multi-step forecasting models.
![end_least](https://github.com/MakenaJones/mental_health_searches/blob/main/images/least_mse_diff_2021-01-01.jpeg)

We fitted SARIMAX and Greykite on data before COVID-19 and checked the difference in the forecast and actual values for the beginning of the COVID-19 restrictions. SARIMAX and Greykite had slightly different predictions for the beginning of COVID-19, but they both over-predicted counseling searches.
![SARIMAX](https://github.com/MakenaJones/mental_health_searches/blob/main/images/forecasting_sarima_counselling.jpeg)

Prophet time series modelling, which performed well in predicting some mental health searches, did not perform better by adding COVID-19 restrictions as one-time holidays during the time periods at the beginning, middle and end of the pandemic. 

---
## Conclusions
Regarding the first part of our problem statement, on whether time series models including the various state restrictions create better forecasts for the various mental health related google search terms than models without, the answer is for certain search terms during certain time periods. More specifically:
* Mental health had an increase in searches towards the end of the pandemic in all states (regardless of restrictions in place)
* Forecasting 'counseling' in beginning of pandemic using restrictions as exogenous features improved the performance of both models (multistep/SARIMAX) for most restricted states.
* In the middle of the pandemic, SARIMA and Recursive multi-step models for the most Restricted States were improved when it came to forecasting 'anxiety', 'mental health' and particularly 'depression', when exogenous features were included in the models.
* At the end of the pandemic, forecasting 'depression' and 'counseling' searches benefited from adding COVID Restrictions as exogenous features in both SARIMAX and recursive multi-step forecasting models for least restricted states.

To answer the second part of our problem statement, regarding whether models see the changing search pattern after restrictions were enforced, the answer is only for 'counseling' searches at the beginning of the Covid-19 pandemic. More specifically:
* SARIMAX and Greykite showed the reduction of counseling searches at the beginning of the Covid-19 restrictions compared to normal for this period of time.

---
## Further Study
 If wanting to isolate the effect of the restrictions themselves, we would have to incorporate other factors that were occuring concurrently before, during and after the pandemic into our models.
 Would be interesting to examine if actual patient numbers/calls to health lines/requests for telehealth services changed for counseling/mental health related services during the periods we examined since we were only looking at Google searches. 

---
## Software Requirements

For this project, we imported pandas, os, matplotlib, cv2, albumentations, sklearn, numpy, mlxtend, tensorflow for streamlit app: streamlit, h5py, gdown 