
from blackjack import generate



# n=0
# print("Thorpe")
# genThorpe = generate(number_of_players = 4,number_of_deck = 6 ,algo="Thorpe",round=50000,modelName= "mlp",TrueCount=True)
# Thorpe = genThorpe["Data"]
# winThorpe = genThorpe["Total Wins"]
# export_csv = Thorpe.to_csv (r'C:\Users\AlankHoang\Desktop\python\Blackjack\export_dataframeThorpe.csv', index = None, header=True)


n=0
print("Random")
genRandom = generate(number_of_players = 4,number_of_deck = 6 ,algo=False,round=100000,modelName= "mlp",TrueCount=True)
training = genRandom["Data"]
winRandom = genRandom["Total Wins"]
export_csv = training.to_csv (r'.\export_dataframeTrainingRandom260419.csv', index = False, header=True)

#print("Thorpe and the win rate is {}".format(winThorpe/Thorpe.shape[0]))
print("Random and the win rate is {}".format(winRandom/training.shape[0]))

