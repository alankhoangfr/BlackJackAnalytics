import os
import random
import pandas as pd
import numpy as np
from prediction import finalRecomendation,recomThorp


cards = [
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'
 
    ]

def cal_hand(hand):
	
	combo=[0]
	if len(hand)==1:
		if hand[0] in ["10","J","Q","K"]:
			combo = [10]
			return combo
		elif hand[0] =="A":
			combo = [1,11]
			return combo
		else:
			if isinstance(hand[0], list):
				if len(hand[0])==0:
					combo = [0]
				else:
					combo = hand[0]
			else:
				combo = [int(hand[0])]
			return combo
	for card in hand:

		temp = [comboHand for comboHand in combo if comboHand <= 21]
		for index, comboHand in enumerate(temp):
			if card in ["J","Q","K"]:
				temp[index]=comboHand+10
			elif card =="A":	
				temp[index]=comboHand+1
				if comboHand<=10:
					temp.append(comboHand+10)		
			elif isinstance(card,list)==False:
				#print(card)
				temp[index]=comboHand+int(card)
		combo = list(set(temp))
	combo = [comboHand for comboHand in combo if comboHand <= 21]
	if int(21) in combo:
		combo = [21]
	return combo

def choice(hand,already):

	if len(hand)==1:
		return ["Hit","Stand","Double"]
	if len(cal_hand(hand))==0:
		return "Bust"
	else:
		if int(21) in cal_hand(hand):
			return "BlackJack"
		elif cal_hand(hand[0])==cal_hand(hand[1]) and hand[0]!=0 and already==False:
			return ["Hit","Stand","Split","Double"]
		elif hand[-1]==0:
			return "Finish"
		else: 
			return ["Hit","Stand","Double"]

def choice_option(option,hand,obs_info,algorithm,modelName,TrueCount):
	obs_copy=obs_info.copy()
	columns = ["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","Result"]
	obs = pd.DataFrame(columns=columns)
	lenobs_copy=len(obs_copy)
	while lenobs_copy!=len(columns):
		obs_copy.append("nan")
		lenobs_copy=len(obs_copy)
	obs.loc[0]=obs_copy
	obs=obs.drop(columns=["Player"])
	if algorithm==True:
		if "Split" in option:
			chooseOption = finalRecomendation(obs,hand,modelName,TrueCount)
		else:
			chooseOption = finalRecomendation(obs,False,modelName,TrueCount)
	elif algorithm=="Thorpe":
		if "Split" in option:
			chooseOption = recomThorp(obs,hand)
		else:
			chooseOption = recomThorp(obs,False)
	else:
		ran = np.random.randint(len(option))
		chooseOption = option[ran]
	return chooseOption


def counting_cards(hand,the_count):
	theCount = the_count
	for cards in hand:
		if cards in ["10","J","Q","K","A"]:
			theCount-=1
		elif int(cards)<=6:
			theCount+=1
		else:
			theCount+=0
	return theCount

def shuffle_pack(deck):
	random.shuffle(deck)
	return deck
# percentage [x,y] where x and Y are between 1 and 100 and X<Y
def shuffle_auto(number_of_deck,deck,percentage,theCount):
	cards = [
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A',
        '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'
 
    ]
	total_inital = int(number_of_deck*52)
	rand = random.randrange(percentage[0],percentage[1],1)/100
	if len(deck)<total_inital*rand:
		new_deck = cards*number_of_deck
		shuffle_pack(new_deck)
		theCount = 0
		print("Shuffle")
		return new_deck,theCount
	else:
		return deck,theCount


def dealing_inital(deck,number_of_players):
	theCount = []
	valueDf = pd.DataFrame()
	n=0
	while n<2:
		inital = {}		
		for i in range(1,number_of_players+1,1):
			new_card= deck.pop()
			inital[int(i)]=new_card
			theCount.append(new_card)
		new_card= deck.pop()
		inital["Dealer"]=new_card
		if n==0:
			theCount.append(new_card)
		valueDf=valueDf.append(inital,ignore_index=True,sort=False)		
		n+=1
	valueDf = total(valueDf)
	valueDf = choice_hand(valueDf)
	return valueDf,theCount

def choice_hand(valueDf):
	numberOfCol = len(valueDf.columns)
	choiceOfHand = {}
	choiceOfHand["Dealer"]=0
	for i in range(1,numberOfCol,1):
		choiceOfHand[i]=choice(valueDf[i].values.tolist()[:-1],already=False)
	valueDf=valueDf.append(choiceOfHand,ignore_index=True,sort=False)
	return valueDf


def total(valueDf):
	numberOfCol = len(valueDf.columns)
	total = {}
	dealerHand = valueDf["Dealer"].values.tolist()
	total["Dealer"]=cal_hand([dealerHand[0]])
	for i in range(1,numberOfCol,1):
		total[i]=cal_hand(valueDf[i].values.tolist())
	valueDf=valueDf.append(total,ignore_index=True,sort=False)
	return valueDf






class One_Game:
	def __init__(self, deck, number_of_players,modelName,TrueCount):
		self.deck = deck
		self.number_of_players = number_of_players
		self.modelName=modelName
		self.TrueCount=TrueCount
	def BlackJackOrBust(self,option,playerCol):
		if option =="BlackJack":
			playerCol.append([21])
			playerCol.append("BlackJack")
		elif option =="Bust":
			playerCol.append([0])
			playerCol.append("Bust")
		return playerCol
	

	def start(self,the_count,algorithm):
		splitnumber={}
		for i in range(1,number_of_players+1):
			splitnumber[i]=0
		playerFinalValue = {}
		NNData = {}
		phase_count = dealing_inital(self.deck,self.number_of_players)
		phase = phase_count[0]
		all_cards = phase_count[1]
		history = pd.DataFrame()
		phaseCopy = phase.loc[:, phase.columns != 'Dealer']
		numPlayer = len(phaseCopy)
		i = 0	
		allPlayerColumn = []	
		while i!= numPlayer:
			blackjack = 0
			colList = phaseCopy.columns.values.tolist()
			colList = sorted(list(map(lambda x:float(x),colList)))
			colName = colList[i]	
			phaseCol = phaseCopy[colName].dropna().values.tolist()
			option = phaseCol[-1]
			playerCol = phaseCol[:-2]
			splitcount = 0
			value = phaseCol[-2]
			hand=playerCol[0:2]
			if "A" in playerCol:
				NNData[colName]=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
					counting_cards(hand =all_cards,the_count=0),
					counting_cards(hand =all_cards,the_count=the_count),
					counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]

			else:
				NNData[colName]=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
					counting_cards(hand =all_cards,the_count=0),
					counting_cards(hand =all_cards,the_count=the_count),
						counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
			obsData = NNData[colName]
			if option =="BlackJack":
				playerCol.append(cal_hand(playerCol))
				playerCol.append("BlackJack")
				chooseOption ="Stand"
				blackjack = 1
			while option not in ["BlackJack","Finish","Bust"]:
				#print(hand,colName,playerCol,"alpha")
				if option==["Hit"]:
					chooseOption="Hit"
				elif (algorithm==True)&(len(playerCol)>1):
					chooseOption = choice_option(option=option,hand=hand,obs_info=obsData,algorithm=True,modelName=self.modelName,TrueCount=self.TrueCount)
					#print("algorithm")
				elif (algorithm=="Thorpe")&(len(playerCol)>1):
					#print("Thorpe")
					chooseOption = choice_option(option=option,hand=hand,obs_info=obsData,algorithm="Thorpe",modelName=self.modelName,TrueCount=self.TrueCount)
				elif (algorithm==False)&(len(playerCol)>1):
					chooseOption = choice_option(option=option,hand=hand,obs_info=obsData,algorithm=False,modelName=self.modelName,TrueCount=self.TrueCount)	
					#print("random",chooseOption)
				if chooseOption =="Hit":
					value = cal_hand(playerCol)	
					if "A" in playerCol:
						NNData[colName]=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					else:
						NNData[colName]=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					nextCard=self.deck.pop()
					playerCol.append(nextCard)	
					all_cards.append(nextCard)
					value = cal_hand(playerCol)		
					option = choice(playerCol,already=True)	
					#print(colName,playerCol)
					if option in ["BlackJack","Finish","Bust"]:
						playerCol = self.BlackJackOrBust(option=option,playerCol=playerCol)
						break
					else:
						if cal_hand(playerCol[0])==cal_hand(playerCol[1]):
							hand=playerCol[0:2]
						else:
							hand = False
					if "A" in playerCol:
						obsData=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					else:
						obsData=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]							
				
				elif chooseOption=="Double":
					value = cal_hand(playerCol)
					if "A" in playerCol:
						NNData[colName]=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]	
					else:
						NNData[colName]=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					nextCard=self.deck.pop()
					playerCol.append(nextCard)
					all_cards.append(nextCard)
					value = cal_hand(playerCol)
					option = "Finish"
					playerCol.append(cal_hand(playerCol))
					playerCol.append("Finish")
					if choice(playerCol[:-2],already=True) in ["BlackJack","Finish","Bust"]:
						break
					else:
						if cal_hand(playerCol[0])==cal_hand(playerCol[1]):
							hand=playerCol[0:2]
						else:
							hand = False
					if "A" in playerCol:
						obsData=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					else:
						obsData=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]	
				elif (splitnumber[int(colName)]<=3) & (chooseOption=="Split"):
					splitnumber[int(colName)]+=1
					#print(colName,playerCol,"PlayerCol",splitnumber,"splitnumber")
					
					firstCard = playerCol[0]
					splitCol = [firstCard,cal_hand([firstCard]),["Hit"]]
					splitColDf = pd.DataFrame({int(colName)+0.1*splitnumber[int(colName)]:splitCol})
					phaseCopy = pd.concat([phaseCopy,splitColDf],axis = 1)		
					phaseCopy.drop(labels=colName, axis="columns", inplace=True)
					secondCard = playerCol[1]
					splitCol1 = [secondCard,cal_hand([secondCard]),["Hit"]]
					splitCol1Df = pd.DataFrame({colName:splitCol1})
					phaseCopy = pd.concat([phaseCopy,splitCol1Df],axis = 1)
					playerCol = [secondCard]
					option=["Hit"]
					numPlayer+=1
					hand=False
					#print("split1",splitColDf,splitCol,"solit2",splitCol1Df,splitCol1)
				elif (splitnumber[int(colName)]>3)& (chooseOption=="Split"):
					#print("no split",splitnumber)
					hand=False
					option=["Hit"]
				elif chooseOption =="Stand":
					value = cal_hand(playerCol)
					if "A" in playerCol:
						NNData[colName]=[colName,min(value),1,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					else:
						NNData[colName]=[colName,value[0],0,max(phase["Dealer"].values.tolist()[2]),
							counting_cards(hand =all_cards,the_count=0),
							counting_cards(hand =all_cards,the_count=the_count),
							counting_cards(hand =all_cards,the_count=the_count)/(len(self.deck)/(52))]
					option = "Finish"
					playerCol.append(cal_hand(playerCol))
					playerCol.append("Finish")
				#print(playerCol,"playercol",colName)
			playerFinalValue[colName]=playerCol[-2]
			#print(playerCol)
			playerColDf = pd.DataFrame({colName:playerCol})
			allPlayerColumn.append(playerColDf)
			i+=1
			NNData[colName].append(chooseOption)
			NNData[colName].append(blackjack)
		for players in allPlayerColumn:
			history = pd.concat([history,players], axis=1)
		DealCol = phase["Dealer"].values.tolist()[:-2]
		DealerValue = int(max(cal_hand(DealCol)))
		DealerValueCopy = int(max(cal_hand(DealCol)))
		all_cards.append(DealCol[1])
		while DealerValue<17:
			nextCard=self.deck.pop()		
			all_cards.append(nextCard)
			DealCol.append(nextCard)
			if len(cal_hand(DealCol))>0:
				DealerValue = int(max(cal_hand(DealCol)))
				DealerValueCopy = int(max(cal_hand(DealCol)))
			elif len(cal_hand(DealCol))==0:
				DealerValueCopy=int(0)
				break
		DealCol.append(DealerValueCopy)
		DealColDf = pd.DataFrame({"Dealer":DealCol})
		history =pd.concat([history,DealColDf], axis=1)
		winCount = 0
		NNDataTotal = pd.DataFrame(columns=["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result"])
		for colName,playerValue in playerFinalValue.items():
			player = playerFinalValue[colName]
			#print(player,"player",colName)
			if len(player)==1:
				playervalue= int(player[0])
				if (playervalue==21)&(DealerValueCopy==21):
					#winCount+=1
					NNData[colName].append(int(1))
				elif playervalue==0:
					NNData[colName].append(int(0))
				elif (playervalue == 21) & (playervalue>DealerValueCopy):
					winCount+=1
					NNData[colName].append(int(2))

				elif playervalue>DealerValueCopy:
					winCount+=1
					NNData[colName].append(int(2))

				elif playervalue<DealerValueCopy:
					NNData[colName].append(int(0))

				elif (playervalue==DealerValueCopy)&(playervalue!=0):
					NNData[colName].append(int(1))

			elif len(player)>1:
				playervalue = int(max(player))
				if (playervalue==21)&(DealerValueCopy==21):
					#winCount+=1
					NNData[colName].append(int(1))
				elif playervalue==0:
					NNData[colName].append(int(0))
				elif (playervalue == 21) & (playervalue>DealerValueCopy):
					winCount+=1
					NNData[colName].append(int(2))

				elif playervalue>DealerValueCopy:
					winCount+=1
					NNData[colName].append(int(2))

				elif playervalue<DealerValueCopy:
					NNData[colName].append(int(0))

				elif (playervalue==DealerValueCopy)&(playervalue!=0):
					NNData[colName].append(int(1))

			elif len(player)==0:
				playervalue=0
				NNData[colName].append(int(0))
				
			NNDataTotal.loc[colName]=NNData[colName]
		#print("theCount",the_count)
		runningCount = counting_cards(hand =all_cards,the_count=the_count)	
		trueCount = runningCount/(len(self.deck)/(52))
		theCountRound = counting_cards(hand =all_cards,the_count=0)
		result={}
		result["history"]=history
		result["winCount"]=winCount
		result["theCountRound"]=theCountRound
		result["runningCount"]=runningCount
		result["trueCount"]=trueCount
		result["NNDataTotal"]=NNDataTotal
		result["all_cards"]=all_cards
		return result

#HyperParameters

number_of_players = 4
number_of_deck = 6 

def generate(number_of_players,number_of_deck,algo,round,modelName,TrueCount):
	NNDataTotal = pd.DataFrame(columns=["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result"])
	deck = cards*number_of_deck
	deck = shuffle_pack(deck)
	rounds = 0
	total_wins = 0
	runningCount = 0
	all_cards = []
	while True:
		deck_count = shuffle_auto(number_of_deck=number_of_deck,
			deck = deck,percentage=[60,75],theCount=runningCount)
		deck = deck_count[0]
		theCount = deck_count[1]
		one = One_Game(deck=deck,number_of_players=4,modelName=modelName,TrueCount=TrueCount)
		show = one.start(the_count=theCount,algorithm=algo)
		runningCount=show["runningCount"]
		showNNDataTotal = show["NNDataTotal"]
		showNNDataTotal["StartRunningCount"]=theCount
		NNDataTotal=NNDataTotal.append(show["NNDataTotal"],ignore_index=True,sort=False)
		print(show)
		total_wins+=show["winCount"]
		rounds +=1
		all_cards.extend(show["all_cards"])
		print(rounds)
		if rounds == round:		
			break
	result = {}
	result["Data"]=NNDataTotal
	result["Total Wins"] = total_wins
	result["Deck"]=all_cards
	return result


def generate_fixed_cards(number_of_players,all_cards,algo,modelName,TrueCount):
	NNDataTotal = pd.DataFrame(columns=["Player","ValueOfHand","Ace","DealerValue","theCountRound","runningCount","trueCount","Action","BlackJackWin","Result"])
	total_wins = 0
	theCount = 0
	deck=all_cards
	#print(len(deck))
	while len(deck) > int(number_of_players)*5:
		one = One_Game(deck=deck,number_of_players=4,modelName=modelName,TrueCount=TrueCount)
		show = one.start(the_count=theCount,algorithm=algo)
		runningCount=show["runningCount"]
		theCount=show["runningCount"]
		showNNDataTotal = show["NNDataTotal"]
		showNNDataTotal["StartRunningCount"]=theCount
		NNDataTotal=NNDataTotal.append(show["NNDataTotal"],ignore_index=True,sort=False)
		#print(show)
		total_wins+=show["winCount"]
	result = {}
	result["Data"]=NNDataTotal
	result["Total Wins"] = total_wins

	return result


#generate(number_of_players = 4,number_of_deck = 6 ,algo="Thorpe",round=50,modelName= "randomForestUnbalanced", TrueCount=True)
