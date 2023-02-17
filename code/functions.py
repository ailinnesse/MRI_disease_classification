from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from os import listdir
from tensorflow.image import grayscale_to_rgb
from tensorflow import convert_to_tensor
from tensorflow.io import read_file, decode_jpeg
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize

# Set image size
image_size = 240

def evaluation_plots(history, name_for_title):
    '''
    Plot loss and Accuracy over epochs
    Input: 
    history - history of the fitted model
    name_for_title - str, the name for the title of the plot
    Output: Side-by-side plots for Categorical Crossentropy and Accuracy
    '''
    # Plot Categorical Crossentropy
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.suptitle(name_for_title, fontsize=20)
    ax[0].plot(history.history['loss'], label='Train', color = 'yellowgreen')
    ax[0].plot(history.history['val_loss'], label = 'Test', color = 'maroon')
    ax[0].set_title('Categorical Crossentropy', size = 20)
    ax[0].set_xlabel('# Epochs', size = 20)
    if len(history.history['loss']) < 15:
        ax[0].set_xticks(range(len(history.history['loss'])))
    ax[0].legend()
    
    # Plot Accuracy
    ax[1].plot(history.history['accuracy'], label='Train', color = 'yellowgreen')
    ax[1].plot(history.history['val_accuracy'], label = 'Test', color = 'maroon')
    ax[1].set_title('Accuracy', size = 20)
    ax[1].set_xlabel('# Epochs', size = 20)
    if len(history.history['accuracy']) < 15:
        ax[1].set_xticks(range(len(history.history['accuracy'])))
    ax[1].legend();
    
    
    
def acc_conf_matrix(model, val_data=None, X=None, y=None, class_names_list = None, binary = False):
    '''
    '''
    # Generate Confusion Matrix
    predictions = np.array([])
    labels =  np.array([])
    if val_data == None:
        if binary:
            predictions = (model.predict(X, verbose = False) > 0.5).astype("int32")
            labels = y  
        else:
            predictions = np.concatenate([predictions, model.predict(X, verbose = False).argmax(axis=1)])
            labels = np.concatenate([labels, np.argmax(y, axis=-1)])

    else:
        for x, y in val_data:
            predictions = np.concatenate([predictions, model.predict(x, verbose = False).argmax(axis=1)])
            labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    cm = confusion_matrix(labels=labels, predictions=predictions).numpy()
    # Print Accuracy score
    acc = round(accuracy_score(labels, predictions), 4)
    print(f'Accuracy_score: {acc} \n') 
    # Plot the confusion matrix
    if class_names_list:
        fig, ax = plot_confusion_matrix(cm, class_names=class_names_list)    
    else:
        fig, ax = plot_confusion_matrix(cm, class_names=val_data.class_names)   
    
    
    
def read_grey_images_to_rgb(path, train_test = True):
    '''
    Read Greyscale images from a directory and convert them to RGB
    Input: 
    path - str, the path to the images
    train_test - bool, Default True. Controls the spliting of the data, if False - no split
    Return: X_train, X_val, y_train, y_val - four arrays ready for TensorFlow models (if train_test = True) or X_test, y_test - two arrays ready for predicting
    '''
    X = []
    y = []

    # Labels from the folder names
    for num_label, label in enumerate(listdir(path)):
        # Change each image and append to X and y
        for image in listdir(f'{path}/{label}'):
            if '.jpg' in image:
                img = read_file(f'{path}/{label}/{image}')  
                img = decode_jpeg(img, channels=1)
                img = resize(img,[image_size, image_size])
                img = convert_to_tensor(img[:,:,:1])
                # Make image RGB for pre-trained models
                img = grayscale_to_rgb(img, name=None)
                X.append(img)
                y.append(num_label)
    # For training the model
    if train_test:
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=18, stratify=y) 
        
        # Change for TensorFlow models
        X_train = np.array(X_train, dtype='float32')
        X_val = np.array(X_val, dtype='float32')

        y_train = to_categorical(y_train, num_classes=4, dtype='float32')
        y_val = to_categorical(y_val, num_classes=4, dtype='float32')

        return X_train, X_val, y_train, y_val
    # For testing
    else:
        X_test = np.array(X, dtype='float32')
        y_test = to_categorical(y, num_classes=4, dtype='float32')
        return X_test, y_test
