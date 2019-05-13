import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, log_loss,roc_auc_score,classification_report,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import dump_svmlight_file
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.model_selection import GridSearchCV
import os
thorpDF = pd.read_csv('thorp.csv')
thorpDF.set_index(['PlayerHand'], drop=True)
df = pd.read_csv('export_dataframeTrainingRandom260419.csv')
df=df.drop(columns=["StartRunningCount","Player","BlackJackWin"])
obj_df = df.select_dtypes(include=['object']).copy()
oneHotEncode = pd.get_dummies(obj_df, columns=["Action"])
data = pd.concat([df, oneHotEncode], axis=1)
data_trueCount=data.drop(columns=["theCountRound","runningCount"])
data_RoundCount=data.drop(columns=["trueCount","runningCount"])

nonScaledColumns = ["Ace","Action","Action_Double","Action_Hit","Action_Stand","Result"]


scaler = preprocessing.Normalizer()

scaled_data_trueCount = scaler.fit_transform(data_trueCount[["ValueOfHand","trueCount","DealerValue"]])
scaled_data_trueCount = pd.DataFrame(scaled_data_trueCount,columns =["ValueOfHand","trueCount","DealerValue"])
scaled_data_trueCount = pd.concat([scaled_data_trueCount,data_trueCount[nonScaledColumns]], axis=1)

scaled_data_RoundCount = scaler.fit_transform(data_RoundCount[["ValueOfHand","theCountRound","DealerValue"]])
scaled_data_RoundCount = pd.DataFrame(scaled_data_RoundCount,columns =["ValueOfHand","theCountRound","DealerValue"])
scaled_data_RoundCount = pd.concat([scaled_data_RoundCount,data_RoundCount[nonScaledColumns]], axis=1)

normalizerTrueCount = preprocessing.Normalizer().fit(data_trueCount[["ValueOfHand","trueCount","DealerValue"]]) 
normalizerRoundCount = preprocessing.Normalizer().fit(data_RoundCount[["ValueOfHand","theCountRound","DealerValue"]]) 
def UpSample(data):
    df_minority = data[data.Result==1]
    df_majority1 = data[data.Result==2]
    df_majority = data[data.Result==0]
    number_of_obsMinority = df_majority.shape[0]

    df_majority_upSample = resample(df_minority, replace=True, n_samples=number_of_obsMinority,random_state=123)
    df_majority1_upSample = resample(df_majority1, replace=True, n_samples=number_of_obsMinority,random_state=123)
    # Combine minority class with downsampled majority class
    df_upSample = pd.concat([df_majority_upSample, df_majority1_upSample,df_majority])

    return df_upSample
Y_trainTrueCount = scaled_data_trueCount["Result"]
Y_trainRoundCount = scaled_data_RoundCount["Result"]

X_scaled_data_trueCount_unbalanced= scaled_data_trueCount.drop(columns=["Result","Action"])
X_scaled_data_roundCount_unbalanced = scaled_data_RoundCount.drop(columns=["Result","Action"])

# Unbalance
X_trainTrueCount_ub, X_testTrueCount_ub, y_trainTrueCount_ub, y_testTrueCount_ub = train_test_split(X_scaled_data_trueCount_unbalanced, Y_trainTrueCount, test_size=0.2, random_state=42)
X_trainRoundCount_ub, X_testRoundCount_ub, y_trainRoundCount_ub, y_testRoundCount_ub = train_test_split(X_scaled_data_roundCount_unbalanced, Y_trainRoundCount, test_size=0.2, random_state=42)

#Balance
xy_trainTrueCount = pd.concat([X_trainTrueCount_ub,y_trainTrueCount_ub],axis = 1)
xy_trainTrueCount = UpSample(data =xy_trainTrueCount)
X_trainTrueCount = xy_trainTrueCount.drop(columns=["Result"])
y_trainTrueCount = xy_trainTrueCount["Result"]

xy_trainRoundCount = pd.concat([X_trainRoundCount_ub,y_trainRoundCount_ub],axis = 1)
xy_trainRoundCount = UpSample(data =xy_trainRoundCount)
X_trainRoundCount = xy_trainRoundCount.drop(columns=["Result"])
y_trainRoundCount = xy_trainRoundCount["Result"]

def optimal_class_weights(model,parameters,weight):
    weights = np.linspace(0.05, 0.95, 5) 
    classWeight = [{0: x, 1: 1.0-x} for x in weights]
    para = parameters
    para.update({weight: classWeight})
    for count,i in enumerate(classWeight):
        i.update({2:1-weights[count]})   
    gsc = GridSearchCV(
        estimator=model,
        param_grid=para,
        scoring='f1_weighted',
        cv=3,
        n_jobs = 3
    )
    grid_result = gsc.fit(X_trainTrueCount_ub, y_trainTrueCount_ub)
    print("Best parameters : %s" % grid_result.best_params_)

    # Plot the weights vs f1 score
    dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                           'weight': weights })
    dataz.plot(x='weight')
    return grid_result.best_params_

print("Training all the models")
# Create directory
dirName = 'Models_pickle'
 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

MLPUnbalance = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)
mlpfitUB= MLPUnbalance.fit(X_trainTrueCount_ub, y_trainTrueCount_ub)
preds = MLPUnbalance.predict(X_testTrueCount_ub)

print("Nerual Network done Imbalanaced True Data")

with open("Models_pickle/MLPUnbalance","wb") as f:
    pickle.dump(MLPUnbalance,f)
    
MLPbalance = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)
mlpfit = MLPbalance.fit(X_trainTrueCount, y_trainTrueCount)
preds = MLPbalance.predict(X_testTrueCount_ub)
print("Nerual Network done Imbalanaced Data True Count")

with open("Models_pickle/MLPbalance","wb") as f:
    pickle.dump(MLPbalance,f)
    
GradientBoostingUnBalance = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
gbtrainUB = GradientBoostingUnBalance.fit(X_trainTrueCount_ub, y_trainTrueCount_ub)
print("Gradient Boosting done Imbalanaced Data True Count")

with open("Models_pickle/GradientBoostingUnBalance","wb") as f:
    pickle.dump(GradientBoostingUnBalance,f)
    
GradientBoostingBalance = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
gbtrain = GradientBoostingBalance.fit(X_trainTrueCount, y_trainTrueCount)
print("Gradient Boosting done balanaced Data True Count")


with open("Models_pickle/GradientBoostingBalance","wb") as f:
    pickle.dump(GradientBoostingBalance,f)
    
parameters = {"n_estimators":[100]}
optimal_param = optimal_class_weights(model=RandomForestClassifier(),parameters=parameters,weight="class_weight")
randomForestUnbalanced = RandomForestClassifier(**optimal_param)
randFUB = randomForestUnbalanced.fit(X_trainTrueCount_ub, y_trainTrueCount_ub)
print("Random Forest done imbalanaced Data True Count")

with open("Models_pickle/randomForestUnbalanced","wb") as f:
    pickle.dump(randomForestUnbalanced,f)
    
randomForestbalanced = RandomForestClassifier(n_estimators=100)
randF = randomForestbalanced.fit(X_trainTrueCount, y_trainTrueCount)
print("Random Forest done balanaced Data True Count")


with open("Models_pickle/randomForestbalanced","wb") as f:
    pickle.dump(randomForestbalanced,f)

clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
clf4 = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)

eclf2UB = VotingClassifier(estimators=[
     ('GB', clf1), ('rf', clf2), ('mlp', clf4)],voting='soft')
parameters = {}
optimal_param = optimal_class_weights(model=eclf2UB,parameters=parameters,weight="weights")
parameters = {"estimators":[('GB', clf1), ('rf', clf2), ('mlp', clf4)],"voting":'soft'}
parameters.update(optimal_param)
eclf2UB = VotingClassifier(**parameters)
vote = eclf2UB.fit(X_trainTrueCount_ub, y_trainTrueCount_ub)
preds = eclf2UB.predict(X_testTrueCount_ub)
print("Ensemble model done imbalanaced Data True Count")    

with open("Models_pickle/voteUnBalance","wb") as f:
    pickle.dump(eclf2UB,f)
    
clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)

clf4 = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)

eclf2 = VotingClassifier(estimators=[
     ('GB', clf1), ('rf', clf2),('mlp', clf4)],voting='soft')
vote = eclf2.fit(X_trainTrueCount, y_trainTrueCount)
preds = eclf2.predict(X_testTrueCount_ub)
print("Ensemble model done balanaced Data True Count")  

with open("Models_pickle/vote","wb") as f:
    pickle.dump(eclf2,f)
    
MLPUnbalance = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)
mlpfitUB= MLPUnbalance.fit(X_trainRoundCount_ub, y_trainRoundCount_ub)
preds = MLPUnbalance.predict(X_testRoundCount_ub)
print("Nerual Network done Imbalanaced Round Data")

with open("Models_pickle/MLPUnbalanceRound","wb") as f:
    pickle.dump(MLPUnbalance,f)
    
MLPbalance = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)
mlpfit = MLPbalance.fit(X_trainRoundCount, y_trainRoundCount)
preds = MLPbalance.predict(X_testRoundCount_ub)
print("Nerual Network done Imbalanaced Round Data")

with open("Models_pickle/MLPbalanceRound","wb") as f:
    pickle.dump(MLPbalance,f)
    
GradientBoostingUnBalance = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
gbtrainUB = GradientBoostingUnBalance.fit(X_trainRoundCount_ub, y_trainRoundCount_ub)
preds = GradientBoostingUnBalance.predict(X_testRoundCount_ub)
print("Gradient Boosting done Imbalanaced Data Round Count")

with open("Models_pickle/GradientBoostingUnBalanceRound","wb") as f:
    pickle.dump(GradientBoostingUnBalance,f)
    
GradientBoostingBalance = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
gbtrain = GradientBoostingBalance.fit(X_trainRoundCount, y_trainRoundCount)
preds = GradientBoostingBalance.predict(X_testRoundCount_ub)
print("Gradient Boosting done balanaced Data Round Count")

with open("Models_pickle/GradientBoostingBalanceRound","wb") as f:
    pickle.dump(GradientBoostingBalance,f)

parameters = {"n_estimators":[100]}
optimal_param = optimal_class_weights(model=RandomForestClassifier(),parameters=parameters,weight="class_weight")
randomForestUnbalanced = RandomForestClassifier(**optimal_param)    
randFUB = randomForestUnbalanced.fit(X_trainRoundCount_ub, y_trainRoundCount_ub)
preds = randomForestUnbalanced.predict(X_testRoundCount_ub)
print("Random Forest done imbalanaced Data Round Count")


with open("Models_pickle/randomForestUnbalancedRound","wb") as f:
    pickle.dump(randomForestUnbalanced,f)
    
randomForestbalanced = RandomForestClassifier(n_estimators=100)
randF = randomForestbalanced.fit(X_trainRoundCount, y_trainRoundCount)
preds = randomForestbalanced.predict(X_testRoundCount_ub)


print("Random Forest done balanaced Data Round Count")
with open("Models_pickle/randomForestbalancedRound","wb") as f:
    pickle.dump(randomForestbalanced,f)
    
clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
clf4 = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)

eclf2UB = VotingClassifier(estimators=[
     ('GB', clf1), ('rf', clf2), ('mlp', clf4)],voting='soft')
parameters = {}
optimal_param = optimal_class_weights(model=eclf2UB,parameters=parameters,weight="weights")
parameters = {"estimators":[('GB', clf1), ('rf', clf2), ('mlp', clf4)],"voting":'soft'}
parameters.update(optimal_param)
eclf2UB = VotingClassifier(**parameters)
vote = eclf2UB.fit(X_trainRoundCount_ub, y_trainRoundCount_ub)
preds = eclf2UB.predict(X_testRoundCount_ub)

print("Ensemble model done imbalanaced Data Round Count")
with open("Models_pickle/voteUnBalanceRound","wb") as f:
    pickle.dump(eclf2UB,f)
    
clf1 = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, max_features=7, max_depth = 2, random_state = 0)
clf2 = RandomForestClassifier(n_estimators=100, random_state=1)

clf4 = MLPClassifier(hidden_layer_sizes = (50,100,), max_iter=1000, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.2,momentum=0.7,batch_size=1000)

eclf2 = VotingClassifier(estimators=[
     ('GB', clf1), ('rf', clf2),('mlp', clf4)],voting='soft')

vote = eclf2.fit(X_trainRoundCount, y_trainRoundCount)
preds = eclf2.predict(X_testRoundCount_ub)
print("Ensemble model done balanaced Data Round Count")

with open("Models_pickle/voteRound","wb") as f:
    pickle.dump(eclf2,f)