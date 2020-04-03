import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def getDis(pointX,pointY,lineX1,lineX2):
    #点到直线距离
    a=0
    b=lineX1-lineX2
    c=0
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis

datasets_X = []
datasets_Y = []
fr = open(r'E:\pubg_python\AI\ai_predict_juesaiquan\dataset3.txt')#===================================================
lines = fr.read()
data1 = eval(lines)
print(len(data1))

for i in data1:
    datasets_X.append(i[:-1])
    datasets_Y.append(i[-1:][0])

for i in datasets_X:
    dis1 = getDis(i[2],i[3],i[0],i[1])
    dis2 = getDis(i[4],i[5],i[0],i[1])
    dis3 = getDis(i[6],i[7],i[0],i[1])
    i.append(dis1)
    i.append(dis2)
    i.append(dis3)
datasets_X = np.array(datasets_X)

datasets_Y = np.array(datasets_Y)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(datasets_X,datasets_Y,test_size=0.1)

clf = DecisionTreeRegressor(random_state=0)
rfc = RandomForestRegressor(random_state=0,n_estimators=100)

clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)

score_c = clf.score(Xtest,Ytest)
score_r = rfc.score(Xtest,Ytest)

print("single Tree:{}".format(score_c),
"Random Forest:{}".format(score_r))

data_draw = []
for i in range(78):
    y_pre = rfc.predict(Xtest[i]).tolist()[0]
    y_true = Ytest[i].tolist()
    data_draw.append([y_true,y_pre])

'''import sys
sys.path.append(r'E:\比赛')
import player_data
xo = player_data.Telemetry_draw(r'E:\比赛\多场1')
xo.draw_ai(data_draw)'''