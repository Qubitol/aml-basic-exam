# Project for exam *Applied Machine Learning - Basic*

**Daniele Massaro**, PhD student in Physics (36° cycle)  
*Alma Mater Studiorum - Università di Bologna*

---

# The MNIST project
The [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology database) is a database of handwritten digits, widely used by the machine learning community to evaluate the capabilities of their algorithms.

> If it doesn't work on MNIST, it won't work at all.  
> Well, if it does work on MNIST, it may still fail on others.  
>   -- *Unknown author*


It was developed by Yann LeCun, Corinna Cortes and Christopher Burges.  
The MNIST database contains 60000 training images and 10000 testing images, which were taken from various scanned documents.
They have been normalised to fit 28x28 pixel bounding box and centered.

Excellent results achieve a prediction error of less than 1%.
State-of-the-art prediction error of approximately 0.2% can be achieved with large Convolutional Neural Networks.

# Method
## The dataset
The database is made of four different datasets:
* [`train-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes);
* [`train-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set labels (28881 bytes);
* [`t10k-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):  test set images (1648877 bytes);
* [`t10k-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):  test set labels (4542 bytes).

The dataset is in binary format.
There exist a ready-to-use `.csv` version, made by Joseph Redmon and available [here](https://pjreddie.com/projects/mnist-in-csv/):
* [`mnist_train.csv`](https://pjreddie.com/media/files/mnist_train.csv);
* [`mnist_test.csv`](https://pjreddie.com/media/files/mnist_test.csv).

Each row consists of 785 values: the first one is the label (an integer from 0 to 9) and the others are the grey-scaled pixels of a 28x28 image (integers ranging from 0 to 255).  
We veryfied the dataset is well-balanced: the class distribution is homogeneous.
This justifies further use of `Kfold` over `StratifiedKFold` and of the *accuracy* score as the main performance metric. 

## Data preparation and feature selection
Given the high dimensionality of the problem and the very high number of data, we decided to use 1/4 of the data and to perform a dimensionality reduction using *Principal Component Analysis* (keeping at least 90% of the explained variance).  
This allowed to speed up the training and the fine-tuning of the various models, at the price of a worse accuracy.  
**The main objective of this work was to show the basic machine learning analysis pipeline, rather than obtaining a striking value of accuracy**.

## Models comparison
This problem is an example of multiclass classification.
The classes are the integer numbers from 0 to 9.
Each hand-written digit is associated to one of them.  
We tried the following models:
* Logistic Regression (LR);
* Linear Discriminant Analysis (LDA);
* k-Nearest Neighbors (KNN);
* Classification and Regression Trees (CART);
* Naive Bayes (NB);
* Linear Support Vector Machines (LSVM);
* Random Forest Classifier (RFC).

For each one of them, we plotted the learning curves, which can be found in `plots/<model_acronym>_learning_curves.pdf`.
We computed the performance metrics *accuracy*, *(weighted) area under ROC curve (AUC)*, *(weighted) F1-score* and the *confusion matrix*.
The values of the computed metrics for each model are saved in the files `cv_metrics/<model_acronym>.npz`.
The algorithms comparison on the basis of the computed metrics is shown in `plots/algorithms_comparison_1.pdf` and `plots/algorithms_comparison_2.pdf`, while the confusion matrices are shown in `plots/confusion_matrices.pdf`.

## Models fine-tuning
We chose to fine-tune the following models:
* KNN;
* RFC.

We choose the accuracy to be the main performance metric for model assessment.
The fine-tuned models are saved in the directory `models/`, ready to be loaded with the `joblib` library.

## Final results
### k-Nearest Neighbors
The accuracy of KNN on the test set is: %.

### Random Forest Classifier
The accuracy of RFC on the test set is: %.
