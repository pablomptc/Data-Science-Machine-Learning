#PERCEPTRON

# libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score



# PART 1: PERCEPTRON
# load iris dataset and divide the data

def perceptron ():
    iris = datasets.load_iris ()
    X =  iris.data[:,[2, 3]]
    y = iris.target



    # divide into training and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 1, stratify = y)



    # standardization
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler ()
    sc.fit (X_train)
    X_train_std = sc.transform (X_train)
    X_test_std = sc.transform (X_test)



    # training: it trains the data
    ppn = Perceptron (eta0 = 0.1, random_state = 1)
    ppn.fit (X_train_std, y_train)



    # prediction: it applies the algorithm to test data
    y_pred = ppn.predict (X_test_std)



    # count the misclassified data
    print ("Misclassified samples: %d" % (y_test != y_pred).sum())



    # accuracy
    print("Accuracy: %.2f" % accuracy_score (y_test, y_pred))



# PART 2: REPRESENTS THE CLASSIFICATION IN A GRAPH
def plot_decision_regions(x,y,classifier, test_idx=None, resolution = 0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])



    # represents the decision surface
    x1_min, x1_max = x[:,0].min() - 1, x[:,0].max() + 1
    x2_min, x2_max = x[:,1].min() - 1, x[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict (np.array ([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape (xx1.shape)
    plt.contourf (xx1,xx2,Z, alpha = 0.3, cmap = cmap)
    plt.xlim (xx1.min(), xx1.max())
    plt.ylim (xx2.min(), xx2.max())



    # represents the flowers depending on the petal width and petal lenght
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl,0], y=x[y == cl, 1],
                   alpha = 0.8, c=colors[idx],
                   marker = markers[idx], label = cl,
                   edgecolor = 'black')



def representation ():
    x_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train,y_test))
    plot_decision_regions(x = x_combined_std, y=y_combined, classifier = ppn, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
