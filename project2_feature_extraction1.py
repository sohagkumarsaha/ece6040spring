# ECE 6040 Project 2  
# Group Member Name: Yang Zheng, Sohag Kumar Saha, Nabil Bin Shahadat Shuva, Ejikeme Amako

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

# Import the necessary libraries for SVM
from sklearn.svm import SVC

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report


# Font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "cm"

# Define get files function
def get_files(filedir):
    files = os.listdir(filedir)
    files.sort()
    return files


# Define Specifity 
def specificity_multi_class(cm):
    specificity = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

# Define classification report option function
def display_metrics(y_true, y_pred, classifier_name):
    print(f"Performance Metrics for {classifier_name}:")
    print(classification_report(y_true, y_pred))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    
    cm = confusion_matrix(y_true, y_pred)
    specificity = specificity_multi_class(cm)
    print(f"Specificity: {specificity}\n")


# Define first feature extraction of training set function
def FeatureExtractorTrain1():
    train_dir = './Project2data/train/'
    train_files = get_files(train_dir)
    x = []
    for n in range(2400):
        filenum_for_plot = n
        file_to_use = train_files[filenum_for_plot]
        rate, signal = wav.read(filename=train_dir + file_to_use)

        time_per_chunk = 0.1
        nperseg = int(rate*time_per_chunk)
        f, t, S = sig.spectrogram(x=signal, fs=rate, nperseg=nperseg, noverlap=0)
        #print(S.shape) #(801, 50)

        fm = []
        for n in range(801):
            fmaximum = np.argmax(S[n,:])
            fm = np.append(fm, fmaximum)
        #print(fm.shape) #(801,)
        x.append(fm)
    Xtrain = np.array(x)
    #print(Xtrain.shape) #(2400, 801)

    y = []
    for n in range(30):
        for m in range(80):
            y.append(n)
    Ytrain = np.array(y)
    #print(Ytrain.shape) #(2400, 1)
    return Xtrain, Ytrain

# Define second feature extraction of training set function
def FeatureExtractorTrain2():
    train_dir = './Project2data/train/'
    train_files = get_files(train_dir)
    x = []
    for n in range(2400):
        filenum_for_plot = n
        file_to_use = train_files[filenum_for_plot]
        rate, signal = wav.read(filename=train_dir + file_to_use)

        time_per_chunk = 0.2
        nperseg = int(rate*time_per_chunk)
        f, t, S = sig.spectrogram(x=signal, fs=rate, nperseg=nperseg, noverlap=5)
        #print(S.shape) #(1601, 25)

        fm = []
        for n in range(1601):
            fmaximum = np.argmax(S[n,:])
            fm = np.append(fm, fmaximum)
        #print(fm.shape) #(1601,)
        x.append(fm)
    Xtrain = np.array(x)
    #print(Xtrain.shape) #(2400, 1601)

    y = []
    for n in range(30):
        for m in range(80):
            y.append(n)
    Ytrain = np.array(y)
    #print(Ytrain.shape) #(2400, 1)
    return Xtrain, Ytrain


# Define first feature extraction of testing set function
def FeatureExtractorTest1():
    train_dir = './Project2data/test/'
    train_files = get_files(train_dir)
    x = []
    for n in range(600):
        filenum_for_plot = n
        file_to_use = train_files[filenum_for_plot]
        rate, signal = wav.read(filename=train_dir + file_to_use)

        time_per_chunk = 0.1
        nperseg = int(rate*time_per_chunk)
        f, t, S = sig.spectrogram(x=signal, fs=rate, nperseg=nperseg, noverlap=0)
        #print(S.shape) #(801, 50)
        fm = []
        for n in range(801):
            fmaximum = np.argmax(S[n,:])
            fm = np.append(fm, fmaximum)
        #print(fm.shape) #(801,)
        x.append(fm)
    Xtest = np.array(x)
    #print(Xtest.shape) #(600, 801)

    y = []
    for n in range(30):
        for m in range(20):
            y.append(n)
    Ytest = np.array(y)
    #print(Ytest.shape) #(600, 1)
    return Xtest, Ytest

# Define second feature extraction of testing set function
def FeatureExtractorTest2():
    train_dir = './Project2data/test/'
    train_files = get_files(train_dir)
    x = []
    for n in range(600):
        filenum_for_plot = n
        file_to_use = train_files[filenum_for_plot]
        rate, signal = wav.read(filename=train_dir + file_to_use)

        time_per_chunk = 0.2
        nperseg = int(rate*time_per_chunk)
        f, t, S = sig.spectrogram(x=signal, fs=rate, nperseg=nperseg, noverlap=5)
        #print(S.shape) #(1601, 25)
        fm = []
        for n in range(1601):
            fmaximum = np.argmax(S[n,:])
            fm = np.append(fm, fmaximum)
        #print(fm.shape) #(1601,)
        x.append(fm)
    Xtest = np.array(x)
    #print(Xtest.shape) #(600, 1601)

    y = []
    for n in range(30):
        for m in range(20):
            y.append(n)
    Ytest = np.array(y)
    #print(Ytest.shape) #(600, 1)
    return Xtest, Ytest

# Define first classifier: Random Forest Classifier
def RFC():
    # First Feature extraction of training set
    Xtrain, Ytrain = FeatureExtractorTrain1()
    Xtest, Ytest = FeatureExtractorTest1()

    ## Second Feature extraction of training set
    #Xtrain, Ytrain = FeatureExtractorTrain2()
    #Xtest, Ytest = FeatureExtractorTest2()

    # Apply random forest classification
    rf = RandomForestClassifier()

    # Fit the random search object to the data
    rf.fit(Xtrain, Ytrain)

    # Generate predirctions with test set
    y_pred = rf.predict(Xtest)

    # Calculate the accuracy and print accuracy
    accuracy = accuracy_score(Ytest, y_pred)
    print("Accuracy:", accuracy)

    # Display Metrics
    display_metrics(Ytest, y_pred, "Random Forest Classifier")    
    
    # Create the confusion matrix
    cm = confusion_matrix(Ytest, y_pred)

    # Plot the confusion matrix
    #ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #plt.title("Confusion Matrix (Random Forest Classifier)")
    #plt.show()
    
    
    fig, ax = plt.subplots(figsize=(12,12))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(30)).plot(ax=ax)
    plt.title("Confusion Matrix (Random Forest Classifier)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('1confusion_matrix_rfc_first_ftext.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    

# Define second classifier: LDA
def LDA():
    # First Feature extraction of training set
    Xtrain, Ytrain = FeatureExtractorTrain1()
    Xtest, Ytest = FeatureExtractorTest1()

    ## Second Feature extraction of training set
    #Xtrain, Ytrain = FeatureExtractorTrain2()
    #Xtest, Ytest = FeatureExtractorTest2()

    # Apply random forest classification
    lda = LinearDiscriminantAnalysis()

    # Fit the random search object to the data
    lda.fit(Xtrain, Ytrain)

    # Generate predirctions with test set
    y_pred = lda.predict(Xtest)

    # Calculate the accuracy and print accuracy
    accuracy = accuracy_score(Ytest, y_pred)
    print("Accuracy:", accuracy)

    
    # Display Metrics
    display_metrics(Ytest, y_pred, "Linear Discriminant Analysis")
    
    # Create the confusion matrix
    cm = confusion_matrix(Ytest, y_pred)

    # Plot the confusion matrix
    #ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    #plt.title("Confusion Matrix (Linear Discriminant Analysis)")
    #plt.show()
    

    fig, ax = plt.subplots(figsize=(12,12))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(30)).plot(ax=ax)
    plt.title("Confusion Matrix (Linear Discriminant Analysis)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('2confusion_matrix_LDA_first_ftext.pdf', dpi=300, bbox_inches='tight')
    plt.show()
       

        
# Define the SVM classifier function
def SVM():
    # First Feature extraction of training set
    Xtrain, Ytrain = FeatureExtractorTrain1()
    Xtest, Ytest = FeatureExtractorTest1()

    ## Second Feature extraction of training set
    #Xtrain, Ytrain = FeatureExtractorTrain2()
    #Xtest, Ytest = FeatureExtractorTest2()

    # Apply SVM classification
    svm = SVC()

    # Fit the SVM model to the data
    svm.fit(Xtrain, Ytrain)

    # Generate predictions with the test set
    y_pred = svm.predict(Xtest)

    # Calculate the accuracy and print the accuracy
    accuracy = accuracy_score(Ytest, y_pred)
    print("Accuracy:", accuracy)
    
    # Display Metrics
    display_metrics(Ytest, y_pred, "Support Vector Machine")

    # Create the confusion matrix
    cm = confusion_matrix(Ytest, y_pred)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12,12))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(30)).plot(ax=ax)
    plt.title("Confusion Matrix (Support Vector Machine)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('3confusion_matrix_SVM_first_ftext.pdf', dpi=300, bbox_inches='tight')
    plt.show()
        

# Define main function
def main():
    ## Use Random Forest Classfier with first feature extraction
    print("Run Random Forest Classifier with First Feature Extraction:")
    print("Running......")
    RFC()

    ## Use LDA with first feature extraction
    print("Run Linear Discriminant Analysis with First Feature Extraction:")
    print("Running......")
    LDA()
    
    
    ## Use SVM with first feature extraction
    print("Run Support Vector Machine with First Feature Extraction:")
    print("Running......")
    SVM()

    # Use Random Forest Classfier with second feature extraction
    #print("Run Random Forest Classifier with Second Feature Extraction:")
    #print("Running......")
    #RFC()

    # Use LDA with second feature extraction
    #print("Run Linear Discriminant Analysis with Second Feature Extraction:")
    #print("Running......")
    #LDA()
    
    
    # Use SVM with second feature extraction
    #print("Run Support Vector Machine with Second Feature Extraction:")
    #print("Running......")
    #SVM()


# Call main function
if __name__ == '__main__':
    main()
