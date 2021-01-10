import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import time
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

t1 = time.time()

data = pd.read_csv("MixEyeGaze.csv")
#print(data.head())

le = preprocessing.LabelEncoder()

left_eye_left_white = le.fit_transform(list(data["left_eye_left_white"]))
left_eye_right_white = le.fit_transform(list(data["left_eye_right_white"]))
left_eye_up_white = le.fit_transform(list(data["left_eye_up_white"]))
left_eye_down_white = le.fit_transform(list(data["left_eye_down_white"]))
right_eye_left_white = le.fit_transform(list(data["right_eye_left_white"]))
right_eye_right_white = le.fit_transform(list(data["right_eye_right_white"]))
right_eye_up_white = le.fit_transform(list(data["right_eye_up_white"]))
right_eye_down_white = le.fit_transform(list(data["right_eye_down_white"]))
#up_ratio = le.fit_transform(list(data["up_ratio"]))
#modified_up_ratio = le.fit_transform(list(data["modified_up_ratio"]))
#mix_ratio = le.fit_transform(list(data["mix_ratio"]))
gaze_lable = le.fit_transform(list(data["gaze_lable"]))
#print(lable)

predict = "gaze_lable"
X = list(zip(left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white,right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white))
#print(X)
#X = preprocessing.normalize(X)
X = preprocessing.scale(X)
#X = list(zip(left_eye_left_white,left_eye_right_white,right_eye_left_white,right_eye_right_white,left_ratio,right_ratio,up_ratio,modified_up_ratio,mix_ratio))
Y = list(gaze_lable)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
"""
datams = pd.read_csv("KNOWN13GP.csv")
#print(data.head())

le = preprocessing.LabelEncoder()

left_eye_left_white = le.fit_transform(list(datams["left_eye_left_white"]))
left_eye_right_white = le.fit_transform(list(datams["left_eye_right_white"]))
left_eye_up_white = le.fit_transform(list(datams["left_eye_up_white"]))
left_eye_down_white = le.fit_transform(list(datams["left_eye_down_white"]))
right_eye_left_white = le.fit_transform(list(datams["right_eye_left_white"]))
right_eye_right_white = le.fit_transform(list(datams["right_eye_right_white"]))
right_eye_up_white = le.fit_transform(list(datams["right_eye_up_white"]))
right_eye_down_white = le.fit_transform(list(datams["right_eye_down_white"]))
#up_ratio = le.fit_transform(list(data["up_ratio"]))
#modified_up_ratio = le.fit_transform(list(data["modified_up_ratio"]))
#mix_ratio = le.fit_transform(list(data["mix_ratio"]))
gaze_lable = le.fit_transform(list(datams["gaze_lable"]))
#print(lable)

predictms = "gaze_lable"
X_ms = list(zip(left_eye_left_white,left_eye_right_white,left_eye_up_white,left_eye_down_white,right_eye_left_white,right_eye_right_white,right_eye_up_white,right_eye_down_white))
#X = list(zip(left_eye_left_white,left_eye_right_white,right_eye_left_white,right_eye_right_white,left_ratio,right_ratio,up_ratio,modified_up_ratio,mix_ratio))
#X_ms = preprocessing.normalize(X_ms)
X_ms = preprocessing.scale(X_ms)
Y_ms = list(gaze_lable)

x_train_ms, x_test_ms, y_train_ms, y_test_ms = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)
#print(x_train,y_test)
"""
#print(x_test)
#print(y_test)
knn_model = KNeighborsClassifier(n_neighbors = 5)       #K=7 for daylight
knn_model.fit(x_train,y_train)
knn_acc = knn_model.score(x_test,y_test)

svm_model = SVC()
svm_model.fit(x_train,y_train)
svm_acc = svm_model.score(x_test,y_test)




lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)
lr_acc = svm_model.score(x_test,y_test)
#t1 = time.time()
rf_model = RandomForestClassifier(n_estimators=20)
rf_model.fit(x_test,y_test)
rf_acc = rf_model.score(x_test,y_test)

nb_model = GaussianNB()
nb_model.fit(x_train,y_train)
nb_acc = nb_model.score(x_test,y_test)

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
dt_acc = dt_model.score(x_test,y_test)
 
print("KNN :",knn_acc)
print("SVM :",svm_acc)
print("Logistic Regression :",lr_acc)
print("Random Forest :",rf_acc)
print("Naive Bayes:",nb_acc)
print("Decission Tree",dt_acc)

#print(t2-t1)
#x_val = [(17, 46, 27, 36, 45, 33, 25, 52)]
#x_val2 =[(10,51,23,38,28,59,26,61)]

#predicted = rf_model.predict(x_test)
#print(predicted)


t2 = time.time()
print(t2-t1)
"""

#Prediction Test by Data
names = ["CENTER","DOWN","DOWN-LEFT","DOWN-RIGHT","GP10","GP11","GP12","GP13","LEFT","RIGHT","TOP","TOP-LEFT","TOP-RIGHT"]

for x in range(len(x_test)):
    print("Predicted",names[predicted[x]],"; Actual",names[y_test[x]])



#Cross Validation Score
knn_k = cross_val_score(KNeighborsClassifier(n_neighbors = 5),X,Y,cv=10)
svm_k = cross_val_score(SVC(),X,Y,cv=10)
lr_k = cross_val_score(LogisticRegression(),X,Y,cv=10)
rf_k = cross_val_score(RandomForestClassifier(n_estimators=10),X,Y,cv=10)

    
print(f'KNN : {knn_k}')
print(f'SVM : {svm_k}')
print(f'LR : {lr_k}')
print(f'RF : {rf_k}')
"""