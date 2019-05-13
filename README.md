# Dependency

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score,classification_report,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.model_selection import GridSearchCV
import os

# Generating the Data

If you want to generate the data in the command line use : python generateData.py
otherwise you can look at the export_dataframeTrainingRandom260419.csv

# Training the model

In the command line use : python trainingModel.py and a new folder Models_pickle will be created and inside will be the pickle for all the models using different data

# Generating the games using thope, random and the other models

In the command line use : python generate.py , you must have trained the models before with python trainingModel.py

# Generating the games using thope, random and the other models using the same cards

In the command line use : python generateSameDeck.py , you must have trained the models before with python trainingModel.py

# Results

Open the BlackJackAnlaytics.ipynb in order to have a look at the analysis of the models.
