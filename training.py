import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import pickle

class dataProcessAndTraining:
	def __init__(self, dataFile,round_or_True):
		self.dataFile = dataFile
		self.round_or_True = round_or_True
	def dataProcess(self):
		df = pd.read_csv(self.dataFile)
		# Data Processing
		obj_df = df.select_dtypes(include=['object']).copy()
		oneHotEncode = pd.get_dummies(obj_df, columns=["Action"])

		data = pd.concat([df.drop(columns=["Action"]), oneHotEncode], axis=1)

		data_trueCount = data.drop(columns=["theCountRound","runningCount"])
		data_RoundCount = data.drop(columns=["trueCount","runningCount"])

		nonScaledColumns = ["Ace","Action_Double","Action_Hit","Action_Stand","Result"]
		scaler = preprocessing.Normalizer()

		scaled_data_trueCount = scaler.fit_transform(data_trueCount[["ValueOfHand","trueCount","DealerValue"]])
		scaled_data_trueCount = pd.DataFrame(scaled_data_trueCount,columns =["ValueOfHand","trueCount","DealerValue"])
		scaled_data_trueCount = pd.concat([scaled_data_trueCount,data_trueCount[nonScaledColumns]], axis=1)

		scaled_data_RoundCount = scaler.fit_transform(data_RoundCount[["ValueOfHand","theCountRound","DealerValue"]])
		scaled_data_RoundCount = pd.DataFrame(scaled_data_RoundCount,columns =["ValueOfHand","theCountRound","DealerValue"])
		scaled_data_RoundCount = pd.concat([scaled_data_RoundCount,data_RoundCount[nonScaledColumns]], axis=1)

		normalizerTrueCount = preprocessing.Normalizer().fit(data_trueCount[["ValueOfHand","trueCount","DealerValue"]]) 
		normalizerRoundCount = preprocessing.Normalizer().fit(data_RoundCount[["ValueOfHand","theCountRound","DealerValue"]]) 
		if self.round_or_True =="RoundCount":
			result = {}
			result["normalizer"]=normalizerRoundCount
			result["scaled_data"]=scaled_data_RoundCount
			
		elif self.round_or_True =="TrueCount":
			result = {}
			result["normalizer"]=normalizerTrueCount
			result["scaled_data"]=scaled_data_trueCount
		return result



	def UpSample(self,data):
	    df_minority = data[data.Result==1]
	    df_majority1 = data[data.Result==2]
	    df_majority = data[data.Result==0]
	    number_of_obsMinority = df_majority.shape[0]

	    df_majority_upSample = resample(df_minority, replace=True, n_samples=number_of_obsMinority,random_state=123)
	    df_majority1_upSample = resample(df_majority1, replace=True, n_samples=number_of_obsMinority,random_state=123)
	    # Combine minority class with downsampled majority class
	    df_upSample = pd.concat([df_majority_upSample, df_majority1_upSample,df_majority])

	    return df_upSample
 
	def splitData(self):
		DataFrames = self.dataProcess()
		DataFrames = DataFrames["scaled_data"]
		scaled_data_balanced = self.UpSample(data =DataFrames)
		Y_trainBalance = scaled_data_balanced["Result"]
		X_scaled_data_balanced = scaled_data_balanced.drop(columns=["Result"])
		X_train, X_test,y_train ,y_test  = train_test_split(X_scaled_data_balanced, Y_trainBalance, test_size=0.2, random_state=42)
		result={}
		result["X_train"]=X_train
		result["X_test"]=X_test
		result["y_train"]=y_train
		result["y_test"]=y_test
		return result



