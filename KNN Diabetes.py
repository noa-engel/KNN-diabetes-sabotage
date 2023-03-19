#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:04:28 2023

@author: noaengel
"""
---
title: "Assigment - kNN DIY"
author:
  - name author here - Noa Engel
  - name reviewer here - Reviewer
date: 13-03-2023
output:
   html_notebook:
    toc: true
    toc_depth: 2
---

pip install pandas
pip install tidyverse
pip install googlesheets4
pip install class
pip install caret
import pandas as pd
from pandas import CategoricalDtype
import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Choose a suitable dataset from [this](https://github.com/HAN-M3DM-Data-Mining/assignments/tree/master/datasets) folder and train  your own kNN model. Follow all the steps from the CRISP-DM model.

## Business Understanding
Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. 
When your blood sugar goes up, it signals your pancreas to release insulin. 
Insulin acts like a key to let the blood sugar into your body’s cells for use as energy.
With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. When there isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, that can cause serious health problems, such as heart disease, vision loss, and kidney disease.
There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help. Other things you can do to help:
 -Take medicine as prescribed.
 -Get diabetes self-management education and support.
 -Make and keep health care appointments.
 From: https://www.cdc.gov/diabetes/basics/diabetes.html#:~:text=Diabetes%20is%20a%20chronic%20(long,your%20pancreas%20to%20release%20insulin.

## Data Understanding
url = "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/KNN-diabetes.csv"
rawDF = pd.read_csv(url)
rawDF = pd.read_csv('https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/KNN-diabetes.csv')
rawDF.info()

testDF = rawDF[(rawDF.Insulin !=0) & (rawDF.BMI !=0)&(rawDF.SkinThickness !=0)&(rawDF.BloodPressure !=0)]
#we remove the zeroes in the dataset to make sure that the zeroes are removed in places where they shouldn't or can't be
cleanDF = rawDF.drop(['BloodPressure'], axis=1) #as far as I know does BloodPressure not influence if you have diabetes or not
cleanDF.head()
rawDF['Outcome'] = rawDF['Outcome'].replace([0],'D') #the D stands for a diagnosis of diabetes
rawDF['Outcome'] = rawDF['Outcome'].replace([1],'N') #the N stands fot a diagnosis of no diabetes

## Data Preparation
cntDiag = cleanDF['Outcome'].value_counts()
propDiag = cleanDF['Outcome'].value_counts(normalize=False)
cntDiag #the outcome is in this case the same as the diagnosis, so to see if someone has diabetes or not
propDiag 

cleanDF.info()

cleanDF[['Pregnancies', 'Insulin', 'BMI']].describe()


def normalize(x):
  return((x - min(x)) / (max(x) - min(x))) # distance of item value - minimum vector value divided by the range of all vector values

testSet1 = np.arange(1,6)
testSet2 = np.arange(1,6) * [[10]]

print(f'testSet1: {testSet1}\n')
print(f'testSet2: {testSet2}\n')
print(f'Normalized testSet1: {normalize(testSet1)}\n')
print(f'Normalized testSet2: {normalize(testSet2)}\n') 

excluded = ['Outcome'] # list of columns to exclude
X = cleanDF.loc[:, ~cleanDF.columns.isin(excluded)]
X = X.apply(normalize, axis=0)

X[['Pregnancies', 'Glucose', 'Diagnose']].describe()

y = cleanDF['Outcome0']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

## Modeling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) #makes predictions based on the test set

y_pred = knn.predict(X_test)
keepdims = True
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

## Evaluation and Deployment
I think the department should look a bit better into the data they presented, there are a lot of zeroes in columns that should have been a higher number (i.e insulin)

Based on the KNN principle, it is now possible to make predictions with a trainingset. It might be interesting
to determine the confidence interval for this model. I do not know how to do this yet, so if there are any suggestions, please tell me
I removed the column BloodPressure because after some research, I found out that this does not infleunce the diagnosis diabetes
I think the KNN is a very interesting way of categorizing and I am certain that it functions the way it has to

For reviewing, I made 5 small alterations for Hussein to find

reviewer adds suggestions for improving the model
