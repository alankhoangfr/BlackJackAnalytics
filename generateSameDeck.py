import os
import random
import pandas as pd
import numpy as np
from blackjack import generate,generate_fixed_cards

model_comparsion = pd.DataFrame(columns=["Model","Info", "Win Rate","Number of plays","Comment"])
export_csv = model_comparsion.to_csv (r'.\model_comparsion_fixedCards10000.csv', index = None, header=True)


print("Random")
column = ["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result","StartRunningCount"]
obs = pd.DataFrame(columns = column)
obs.to_csv(r'.\Model_Data_testFixed\RandomRun.csv', index = None, header=True)

genTest = generate(number_of_players = 4,number_of_deck = 6 ,algo=False,round=10000,modelName= "MLPUnbalance", TrueCount=True)
test = genTest["Data"]
winTest = genTest["Total Wins"]	
all_cards = genTest["Deck"]
test.to_csv(r'.\Model_Data_testFixed\RandomRun.csv', index = None, header=True)	
model_comparsion = pd.read_csv('model_comparsion_fixedCards10000.csv')
model = {}
model["Model"]="RandomRun"
model["Info"]="4 Players, 6 Decks 5000 Rounds"
model["Win Rate"]=winTest/test.shape[0]
model["Number of plays"]=test.shape[0]
model["Comment"]="None"
model_comparsion=model_comparsion.append(model,ignore_index=True,sort=False)
model_comparsion.to_csv (r'.\model_comparsion_fixedCards10000.csv', index = None, header=True)
cards = pd.DataFrame()
cards["Card"]=all_cards
cards.to_csv(r'.\Model_Data_testFixed\deck.csv', index = None, header=True)



print("Thorpe")
column = ["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result","StartRunningCount"]
obs = pd.DataFrame(columns = column)
obs.to_csv(r'C:\Users\AlankHoang\Desktop\python\Blackjack\Model_Data_testFixed\ThorpeRun.csv', index = None, header=True)
deck = pd.read_csv('Model_Data_testFixed/deck.csv')	
deck=deck["Card"].values.tolist()
genTest = generate_fixed_cards(number_of_players = 4,all_cards=deck ,algo="Thorpe",modelName= "MLPUnbalance", TrueCount=True)
test = genTest["Data"]
winTest = genTest["Total Wins"]
test.to_csv(r'.\Model_Data_testFixed\ThorpeRun.csv',index = None, header=True)		
#print(test)
model_comparsion = pd.read_csv('model_comparsion_fixedCards10000.csv')
model = {}
model["Model"]="ThorpeRun"
model["Info"]="4 Players, Same Cards as the Random Run"
model["Win Rate"]=winTest/test.shape[0]
model["Number of plays"]=test.shape[0]
model["Comment"]="None"
model_comparsion=model_comparsion.append(model,ignore_index=True,sort=False)
model_comparsion.to_csv (r'.\model_comparsion_fixedCardstest.csv', index = None, header=True)


def compare(model_names,number_of_players,TrueCount,saveModel):
	deck = pd.read_csv('Model_Data_testFixed/deck.csv')	
	deck=deck["Card"].values.tolist()
	for models in model_names:
		column = ["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result","StartRunningCount"]
		obs = pd.DataFrame(columns = column)
		obs.to_csv(r'.\Model_Data_testFixed\{}.csv'.format(models), index = None, header=True)
		print("{}".format(str(models)))
		genTest = generate_fixed_cards(number_of_players = number_of_players,all_cards=deck ,algo=True,modelName= models ,TrueCount=TrueCount)
		test = genTest["Data"]
		winTest = genTest["Total Wins"]
		test.to_csv(r'.\Model_Data_testFixed\{}.csv'.format(models), index = None, header=True)	
		model_comparsion = pd.read_csv('{}.csv'.format(saveModel))
		model = {}
		model["Model"]="{}".format(models)
		model["Info"]="{} Players, Same Cards as the Random Run".format(number_of_players)
		model["Win Rate"]=winTest/test.shape[0]
		model["Number of plays"]=test.shape[0]
		if TrueCount==True:
			model["Comment"]="True Count"
		else:
			model["Comment"]="Round Count"
		model_comparsion=model_comparsion.append(model,ignore_index=True,sort=False)
		model_comparsion.to_csv (r'.\{}.csv'.format(saveModel), index = None, header=True)
		deck = pd.read_csv('Model_Data_testFixed/deck.csv')	
		deck=deck["Card"].values.tolist()

model_names = ["MLPUnbalance","MLPbalance","GradientBoostingBalance","GradientBoostingUnBalance","randomForestUnbalanced","randomForestbalanced","vote","voteUnBalance"]
#model_names = ["MLPUnbalance","MLPbalance","GradientBoostingBalance","GradientBoostingUnBalance"]
#model_names =["randomForestUnbalanced","randomForestbalanced","vote","voteUnBalance"]
compare(model_names = model_names,number_of_players = 4, TrueCount=True,saveModel="model_comparsion_fixedCards10000")
model_names_Round = ["MLPUnbalanceRound","MLPbalanceRound","GradientBoostingBalanceRound","GradientBoostingUnBalanceRound","randomForestUnbalancedRound",
"randomForestbalancedRound","voteRound","voteUnBalanceRound"]
#model_names_Round = ["MLPUnbalanceRound","MLPbalanceRound","GradientBoostingBalanceRound","GradientBoostingUnBalanceRound"]
#model_names_Round = ["randomForestUnbalancedRound","randomForestbalancedRound","voteRound","voteUnBalanceRound"]
compare(model_names = model_names_Round,number_of_players = 4, TrueCount=False,saveModel="model_comparsion_fixedCards10000")