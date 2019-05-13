
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
from training import dataProcessAndTraining


model = dataProcessAndTraining(dataFile="export_dataframeTrainingRandom260419.csv",round_or_True="TrueCount")
data = model.dataProcess()
normalizerTrueCount = data["normalizer"]

model = dataProcessAndTraining(dataFile="export_dataframeTrainingRandom260419.csv",round_or_True="RoundCount")
data = model.dataProcess()
normalizerRoundCount = data["normalizer"]

thorpDF = pd.read_csv('thorp.csv')
thorpDF.set_index(['PlayerHand'], drop=True)

model_names=["MLPUnbalance","MLPbalance","GradientBoostingBalance","GradientBoostingUnBalance","randomForestUnbalanced","randomForestbalanced","vote","voteUnBalance",
"MLPUnbalanceRound","MLPbalanceRound","GradientBoostingBalanceRound","GradientBoostingUnBalanceRound","randomForestUnbalancedRound","randomForestbalancedRound",
"voteRound","voteUnBalanceRound"]


def model_selection(model_names):
    MODEL_SELECTION = {}
    for models in model_names:
        with open("Models_pickle/{}".format(models),"rb") as f:
            mod =pickle.load(f)
        MODEL_SELECTION [models] = mod
    return MODEL_SELECTION

MODEL_SELECTION = model_selection(model_names=model_names)


def processObs(observation,TrueCount):
    if TrueCount == True:
        obs = observation
        scaled_columns = ["ValueOfHand", "trueCount","DealerValue"]
        scen_action = pd.DataFrame(columns=["Action"])
        scen_action["Action"] = ["Hit","Stand","Double"]
        oneHotEncode = pd.get_dummies(scen_action, columns=["Action"])
        normalise = obs[["ValueOfHand", "trueCount","DealerValue"]].values.tolist()
        normalise
        scaled_instances =  normalizerTrueCount.transform(normalise)
        scaled_instances[0]
        obs=obs.drop(columns=["theCountRound","runningCount","Result","Action"])
        for count ,col in enumerate(scaled_columns):
            obs[col]=scaled_instances[0][count]
        obs=obs[["ValueOfHand", "trueCount","DealerValue","Ace"]]
        obs=obs.append([obs]*2,ignore_index=True,sort=False)
        obs = pd.concat([obs,oneHotEncode],axis=1)
        return obs
    elif TrueCount==False:
        obs = observation
        scaled_columns = ["ValueOfHand", "theCountRound","DealerValue"]
        scen_action = pd.DataFrame(columns=["Action"])
        scen_action["Action"] = ["Hit","Stand","Double"]
        oneHotEncode = pd.get_dummies(scen_action, columns=["Action"])
        normalise = obs[["ValueOfHand","theCountRound","DealerValue"]].values.tolist()
        scaled_instances =  normalizerRoundCount.transform(normalise)
        scaled_instances[0]
        obs=obs.drop(columns=["trueCount","runningCount","Result","Action"])
        for count ,col in enumerate(scaled_columns):
            obs[col]=scaled_instances[0][count]
        obs=obs[["ValueOfHand", "theCountRound","DealerValue","Ace"]]
        obs=obs.append([obs]*2,ignore_index=True,sort=False)
        obs = pd.concat([obs,oneHotEncode],axis=1)
        return obs


def recomThorp(obs,hand):
    #print(obs,hand)
    if obs["DealerValue"].iloc[0]==11:
        DealerValue = "A"
    else:
        DealerValue="{}".format(obs["DealerValue"].iloc[0])
    if isinstance(hand, list):
        for i in range (0,2,1):
            if hand[i] in ["J","Q","K"]:
                hand[i]=10
        playerHand = "{},{}".format(hand[0],hand[1])
        thorp = thorpDF.loc[thorpDF["PlayerHand"]==playerHand,[DealerValue]]
        thorp=thorp.values[0][0]
        return thorp
    elif hand==False:
        if obs["Ace"].iloc[0]==1:
            if obs["ValueOfHand"].iloc[0]>10:
                playerHand = "{}".format(int(obs["ValueOfHand"].iloc[0]))
            elif obs["ValueOfHand"].iloc[0]==1:
                playerHand = "A,A"
            else:
                playerHand = "A,{}".format(int(obs["ValueOfHand"].iloc[0])-1)
            thorp = thorpDF.loc[thorpDF["PlayerHand"]==playerHand,[DealerValue]]
            thorp=thorp.values[0][0]
        else:
            playerHand = "{}".format(obs["ValueOfHand"].iloc[0])
            thorp = thorpDF.loc[thorpDF["PlayerHand"]==playerHand,[DealerValue]]
            thorp=thorp.values[0][0]
        return thorp



def finalRecomendation(obs,hand,modelName,TrueCount):

    #print(obs,hand,"finalRecomendation")
    if isinstance(hand, list): 

        result = recomThorp(obs,hand)
    elif hand==False:
        recom = ["Hit","Stand","Double"]
        if obs["ValueOfHand"].iloc[0]>=17:
            result = "Stand"
            #print(result,"overide")
            return result
        elif obs["ValueOfHand"].iloc[0]<=11:
            result =recomThorp(obs,hand)
            #print(result,"overide")
            return result
        pObs= processObs(observation = obs,TrueCount=TrueCount)
        recommendation = MODEL_SELECTION[modelName].predict(pObs).tolist()
        if recommendation.count(2)==1:
            result = recom[recommendation.index(2)]
        elif (recommendation.count(1)==1)&(recommendation.count(2)==0):
            result = recom[recommendation.index(1)]
        elif recommendation == [1,0,1]:
            result = "Hit"
        else:
            result = recomThorp(obs,hand)
        
        if (result=="Hit") &  (recomThorp(obs,hand)=="Double"):
            result = "Double"
        elif (result=="Double") &  (recomThorp(obs,hand)=="Hit"):
            result = "Hit"
        #print(result,"asdfsdfsd")
    return result


