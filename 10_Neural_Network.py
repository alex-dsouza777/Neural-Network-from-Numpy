# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:20:08 2021

@author: Alex D'souza
"""

# In this Assignmet we are developing Neural Network from Scratch :-> Using Numpy

#Importing libraries
from sklearn import datasets #For creating dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report 

np.random.seed(0) #Sets predictable numbers list

# Generating dataset

feature_set, label = datasets.make_moons(100,noise=0.05) 

#Visualization of Generated dataset
plt.figure(figsize=(7,4))
plt.scatter(feature_set[:,0],feature_set[:,1],c = label,cmap = plt.cm.winter)
plt.title("Actual Data")
plt.show()

#Reshaping 'label' as One array with 100 arrays of Target Label
labels = label.reshape(-1,1)
print(labels.shape)
print(feature_set.shape)

    
#Defining Activation function for classification
def sigmoid(x):
    return 1 / (1+np.exp(-x))

#Derivation of Sigmoid
def sigmoid_derivative(x):
    return sigmoid(x)* (1-sigmoid(x))



np.random.seed(42) #Here, Sets New predictable numbers list


#Initializing weights
weights = np.random.rand(2,1)

#Initializing Bias
bias = np.random.rand(1)

#Defining Learning Rate
lr = 0.5


#Training Neural Network with Number of Epochs
for epoch in range(10):
    inputs = feature_set #Taking input features
    
    # Forward Propagation <Multipying Input Feature with Weigts Value + Adding Bias Value>
    XW = np.dot(feature_set,weights) + bias
    
    # Applying Sigmoid activation Function which converts the value into range of 0-1
    z = sigmoid(XW)
        
    # Predicted - Actual Target Value
    error = z - labels
    
    dcost_dpred = error
    
    # Cost function of the Loss value i.e. Total Error
    error_value = np.power((z - labels),2)
    
    print("Error: ",error_value.sum())
    
    # For Back Propagation
    
    # Derivative of Predicted value
    dpred_dz = sigmoid_derivative(z)
    
    # Derivative of Predicted value is multiplied with error
    z_delta = dcost_dpred * dpred_dz
    
    # Transpose of Featuer Set
    inputs = feature_set.T
    
    #Updating Weight Formula -> Weight_old - Learnig_Rate * (Derivative_Loss/Derivative_Weight_Old)
    weights = weights - lr * np.dot(inputs ,z_delta)
    
    # Updating Bias value
    for num in z_delta:
        bias = bias - lr * num
        


#Creating a simple Prediction Function

def prediction(x_input):
    value = np.dot(x_input,weights) + bias #<Multipying Input Feature with Weigts Value + Adding Bias Value>
    return sigmoid(value)

# Function for returning Binary Output with Threshold of 0.5

def return_bin(x):
    pred_list = []
    for i in x:
        if i < 0.5:
            pred_list.append(0)
        else:
            pred_list.append(1)
    return np.array(pred_list)


# Class for Model prediction 
class model:
    def predict(x_input):
        
        value = np.dot(x_input,weights) + bias
        a = sigmoid(value)
        lbl_pred = return_bin(a)
        return lbl_pred

X = feature_set # X as Input Features
y = label # y as Target Label

# Predicting all inputs
y_pred = model.predict(X) 

# Checking performace metrics
print("Accuracy Score: ",accuracy_score(y_pred,y))
print("Classification Report: \n",classification_report(y_pred,y))



# Plotting decision boundary

def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    #h = 0.5
    h = lr
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 


plot_decision_boundary(lambda x: model.predict(x))
plt.title("Neural Network_decision_boundary") 
plt.show()


from matplotlib.colors import ListedColormap  
x_set, y_set = X, y 
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = lr))  
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
   plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
plt.title('Neural Network classifier (Whole Data set)')  
plt.legend()  
plt.show()  


