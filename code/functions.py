from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score



def evaluation_plots(history, name_for_title):
    '''
    Plot loss and Recall over epochs
    Input: history of the fitted model, the name for the title
    Output: Side-by-side plots for Categorical Crossentropy and Recall
    '''
    # Plot Categorical Crossentropy
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.suptitle(name_for_title, fontsize=20)
    ax[0].plot(history.history['loss'], label='Train', color = 'yellowgreen')
    ax[0].plot(history.history['val_loss'], label = 'Test', color = 'maroon')
    ax[0].set_title('Categorical Crossentropy', size = 20)
    ax[0].set_xlabel('# Epochs', size = 20)
    ax[0].set_xticks(range(len(history.history['loss'])))
    ax[0].set_ylabel('Categorical Crossentropy', size = 20)
    ax[0].legend()
    # Plot Accuracy
    ax[1].plot(history.history['accuracy'], label='Train', color = 'yellowgreen')
    ax[1].plot(history.history['val_accuracy'], label = 'Test', color = 'maroon')
    ax[1].set_title('Accuracy', size = 20)
    ax[1].set_xlabel('# Epochs', size = 20)
    ax[0].set_xticks(range(len(history.history['accuracy'])))
    ax[1].set_ylabel('Recall', size = 20)
    ax[1].legend();
    
    
    
    
    
def acc_conf_matrix(val_data, model):
    '''
    '''
    # Generate Confusion Matrix
    predictions = np.array([])
    labels =  np.array([])
    for x, y in val_data:
        predictions = np.concatenate([predictions, model.predict(x, verbose = False).argmax(axis=1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
    cm = confusion_matrix(labels=labels, predictions=predictions).numpy()
    # Print Accuracy score
    acc = round(accuracy_score(labels, predictions), 4)
    print(f'Accuracy_score: {acc} \n') 
    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(cm, class_names=val_data.class_names)