# Iris-Flower-Classification
Building a web app using streamlit and deploying it to the cloud with Heroku.

## Table of contents.
1. Introduction
2. Problem statement
3. Data Cleaning and Processing
4. Data Visulaization
5. Building a machine learning model
6. Saving the machine learning model
7. Build the app using python library ‘Streamlit’
8. Deployment via Heroku
9. Conclusion


## Introduction
Iris is the family in the flower which contains the several species such as the iris.setosa,iris.versicolor,iris.virginica,etc.

## Problem Statement 
We have to do classification of the flowers based on the features mentioned.
The data set consists of:
150 samples
3 labels: species of Iris (Iris setosa, Iris virginica and Iris versicolor)
4 features: Sepal length,Sepal width,Petal length,Petal Width in cm

## Building a machine learning model
In this article, I will not go much into detail about building the machine learning model. However, I will just list down the steps needed to create a model.
Import all the required libraries.
Read the dataset.
Data Cleansing.
Understand the data — exploratory data analysis.
Feature selection — Multiple strategies such as correlation matrix, Chi square, ANOVA can be implemented to finalize the features you want to use to build the web app for the prediction.
Separate the input and output features.
Do feature scaling if needed — We can use either standardization (StandardScaler) or normalization (MinMaxScaler) based on our features.
Label Encoding the Data.
Split the data into train and test.
Build the models on the train data-try out multiple models and chose the one with the best accuracy.
Use the trained model to predict on the test data.

## Saving the machine learning model
Let us understand the need of saving a machine learning model.
Suppose, in our example problem statement, I have used JuPyter Notebook for building my model to help me with the predictions on any new data. So, if I want to do the predictions on another day or some other time, I have to run the whole notebook again. Or, if my friends or colleagues want to do the same prediction, then I will have to pass my entire notebook to them for their use.
Hence, instead of running the entire code again , we can just save and load the model whenever and wherever required. It is also called dumping the model as we are storing information such as coefficients and other related data about the model.
It basically indicates re-usability of our code.
There are numerous python packages to save a machine learning model like
. Pickle
. Joblib
However , I have used Pickle module for my task.
So, once we finalize the model with the best accuracy , we can save the model on the disk with the pickle module in python by first importing it.
The back-end code for the data processing and the model building is available in github.

## Build the app using python library ‘Streamlit’
Whenever we think about building a web app, all that comes to our mind are HTML, JavaScript and almost all the companies have designated teams completely comprised of front end web developers. Since data scientists are more concerned about the processing of the data , the back-end development of models which itself is extremely time consuming, it might not be easier for a data scientist to again invest time and energy on front end development.
However, there is this python library called ‘Streamlit’ which comes as relief to a data scientist as Streamlit is one of the easiest ways to build a front end web app for our machine learning projects with simple python scripts. And all these without any HTML or Javascript.
Streamlit is an open source python library similar to a wrapper class.
We all know what a wrapper class is. A simple definition of wrapper class is any class which “wraps” or “encapsulates” the functionality of another class or component. Streamlit behaves in the exact same way. We load the model and pass the features entered by the end user back to our model based on which all the processing is done on the back end and finally we send the result/prediction back to the web app.

Streamlit Link : http://localhost:8501

## Deployment via Heroku
Heroku is a container-based cloud Platform as a Service (PaaS) where we can easily deploy web apps for our end users to use.
What is the need for deployment:
A simple explanation: Till now , my app runs only on my local system. So, if someone else wants to use my app, I would have to forward him/her my notebooks with all the code, then they will need to run the entire code again, save it in their system, and again run it in their local system to use the app .Lengthy process right ?

## Conclusion
In this project we have examined on Iris Flower CLassification Data by performing step by step operation of EDA/VDA. Here on, applying machine learning Logistic Regression Model and Random Forest Classifier Model.
Considering the Best Accuracy Score building a streamlit page then hereon deploying on cloud platform (Heroku).

## Heroku Web App Link :
https://iris-project-divya.herokuapp.com/

