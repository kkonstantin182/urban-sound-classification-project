import matplotlib.pyplot as plt 
import numpy as np
import os
import json
import random
import tensorflow as tf

def save_metrics(history : dict, name : str, path="/content/drive/MyDrive/UNIMI/MachineLearning/UrbanSound/models"):  
    """
    Save metrics (loss and accuracy) for each epoch from the trained model
    to the json file.
    """
    with open(os.path.join(path ,f"{name}.json"), "w") as f: 
      json.dump(history, f)

def print_metrics(history_dict):
  
    """
    Print accuracy and loss on the training set 
    and the validation set on the last epoch.
    """
    length = len(history_dict['loss']) - 1
    print("Training loss and training accuracy: " + str([np.round(loss_acc_extract(history_dict)[0][length - 1], 2), np.round(loss_acc_extract(history_dict)[2][length - 1], 2)]))
    print("Validation loss and validation accuracy: " + str([np.round(loss_acc_extract(history_dict)[1][length - 1], 2), np.round(loss_acc_extract(history_dict)[3][length - 1], 2)]))

def loss_acc_extract(history_dict):
     
    """
    Extract following metrics: 
        -loss on the training set
        -loss on the validation set
        -accuracy on the training set
        -accuracy on the validation set
    """
    loss_values = np.array(history_dict['loss'])
    val_loss_values = np.array(history_dict['val_loss'])
    acc_values = np.array(history_dict['accuracy'])
    val_acc_values = np.array(history_dict['val_accuracy'])
    return loss_values, val_loss_values, acc_values, val_acc_values

def big_loss_plot(histories, legend=[], size=20, xlim=[None, None], ylim=[0, 3.5]):

    """
    Plot loss vs number of epochs
    """
    if not legend:
      legend = [i for i in range(1, len(histories) + 1)]
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel('Epochs', fontsize=size)
    ax.set_ylabel('Loss', fontsize=size)
    ax.tick_params(axis='both', which='major', labelsize=size)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for i in range(len(histories)):
        loss = loss_acc_extract(histories[i])[0]
        val_loss = loss_acc_extract(histories[i])[1]
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        ax.plot(range(1, len(loss) + 1), loss, linestyle='solid', color=(r/255.0, g/255.0, b/255.0), label=f"TS, {legend[i]}")
        ax.plot(range(1, len(val_loss) + 1), val_loss, linestyle='dotted', color=(r/255.0, g/255.0, b/255.0), label=f"VS, {legend[i]}")
    ax.legend(prop={'size': size - 4 })

def big_accuracy_plot(histories, legend=[], size=20, ylim=[0, 1]):

    """
    Plot accuracy plot vs number of epochs
    """
    if not legend:
      legend = [i for i in range(1, len(histories) + 1)]
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel('Epochs', fontsize=size)
    ax.set_ylabel('Accuracy', fontsize=size)
    ax.tick_params(axis='both', which='major', labelsize=size)
    ax.set_ylim(ylim)
    for i in range(len(histories)):
        acc = loss_acc_extract(histories[i])[2]
        val_acc = loss_acc_extract(histories[i])[3]
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        ax.plot(range(1, len(acc) + 1), acc, linestyle='solid', color=(r/255.0, g/255.0, b/255.0), label=f"TS, {legend[i]}")
        ax.plot(range(1, len(val_acc) + 1), val_acc, linestyle='dotted', color=(r/255.0, g/255.0, b/255.0), label=f"VS, {legend[i]}")
    ax.legend(prop={'size': size - 4 })

def loss_vs_param_plot(histories, 
                       par, 
                       par_name="parameter", 
                       size=20, 
                       line=False):

    """
    Plot loss plot vs parameter
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlabel(f'{par_name}', fontsize=size)
    ax.set_ylabel('Loss', fontsize=size)
    ax.tick_params(axis='both', which='major', labelsize=size)
    loss_list = []
    val_loss_list = []
    for i in range(len(histories)):
        loss = loss_acc_extract(histories[i])[0]
        val_loss = loss_acc_extract(histories[i])[1]
        length = len(loss)
        loss_list.append(loss[length - 1])
        val_loss_list.append(val_loss[length - 1])
    ax.scatter(par, loss_list, c="k", label="TS", s=size*5)
    ax.scatter(par, val_loss_list, c="r", label="VS", s=size*5)
    if line==True:
      ax.plot(par, loss_list, "k--")
      ax.plot(par, val_loss_list, "r--")
    ax.legend(prop={'size': size - 4 })


def reset_random_seeds():

    """
    Ensures reproducibility of the results.
    The solutin was taken from:
    https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
    """
    os.environ['PYTHONHASHSEED']=str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

def fold_evaluate(model, X, y):

    """
    Evaluate accuracy on each fold using built-in method "model.evalute"
    on each fold and return the avrage accuracy with standard deviation.
    """
    print("Evaluate the model on the test set")
    acc_list = []
    for i in range(len(X)):
      res = model.evaluate(X[i], y[i])
      acc_list.append(res[1])
    return("Average test accuracy: " + str(round(np.mean(acc_list), 2)) + u"\u00B1" + str(round(np.std(acc_list), 2)))
    
    
    
def acc_loss_plot(history, size=26, kind="train"):
    color1 = "#FF5733"
    color2 = "#335DFF"
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.set_xlabel('Epochs', fontsize=size)
    ax1.set_ylabel('Accuracy', fontsize=size, color=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=size, color=color2)
    ax1.tick_params(axis='both', which='major', labelsize=size)
    ax2.tick_params(axis='both', which='major', labelsize=size)
    ax1.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)
    loss = loss_acc_extract(history)[0]
    val_loss = loss_acc_extract(history)[1]
    acc = loss_acc_extract(history)[2]
    val_acc = loss_acc_extract(history)[3]
    if kind == "train":
        ax1.plot(range(1, len(acc) + 1), acc, linestyle='solid', linewidth=4, color=color1, label="Accuracy, TS")
        ax2.plot(range(1, len(loss) + 1),loss, linestyle='dotted', linewidth=4, color=color2, label="Loss, TS")
    else:
        ax1.plot(range(1, len(val_acc) + 1), val_acc, linestyle='solid', linewidth=4, color=color1, label="Accuracy, VS")
        ax2.plot(range(1, len(val_loss) + 1), val_loss, linestyle='dotted', linewidth=4, color=color2, label="Loss, VS")
    ax1.legend(prop={'size': size})
    ax2.legend(prop={'size': size})
    ax1.grid(color=color1)
    ax2.grid(color=color2)