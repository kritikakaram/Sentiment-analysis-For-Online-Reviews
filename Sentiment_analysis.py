from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
model = TfidfVectorizer(stop_words=stopwords.words('english'))

#read the reviews and their polarities from a given file

def trainingloadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.split('\t')  
        reviews.append(review.lower())    
        labels.append(rating)
    f.close()
    return reviews,labels

train_reviews,train_labels=trainingloadData('training_data.txt')

def testingloadData(name):
    reviews=[]
    f=open(name,encoding='utf-8')
    for line in f:
        review=line.strip()
        reviews.append(review.lower())
    f.close()
    return reviews

################################## TRAINING FILE INPUT###########################

test_reviews=testingloadData('testing.txt')

##################################################################################
for i in range(len(test_reviews)):
    test_reviews[i]=re.sub(" +"," ",test_reviews[i])
    test_reviews[i]=re.sub("\t","",test_reviews[i])

# Removing all the special characters
for i in range(len(test_reviews)):
    test_reviews[i]=re.sub("[^A-Za-z0-9]"," ",test_reviews[i])

# Converting Upper cases to lower cases
for i in range(len(test_reviews)):
    test_reviews[i]=test_reviews[i].lower()

model.fit(train_reviews)    
X_train=model.transform(train_reviews)
y_train=train_labels

X_test=model.transform(test_reviews)
#y_test=test_labels




model1 = RandomForestClassifier(n_jobs=-1)
model2 = MultinomialNB()
model3=LogisticRegression(n_jobs=-1)

predictors=[('nb',model2),('lreg',model3),('RF',model1)]

VT=VotingClassifier(predictors)
#======================================================================================

model2 = MultinomialNB()
model2.fit(X_train, y_train)

#=======================================================================================

#build the parameter grid
DT_grid = [{'max_depth': [5,6,7,8,9,10,11,12,None],'criterion':['gini','entropy']}]

#build a grid search to find the best parameters
gridsearchDT  = GridSearchCV(model1, DT_grid, cv=5)

#run the grid search
gridsearchDT.fit(X_train,y_train)
best_accdt= gridsearchDT.best_score_
best_paramdt= gridsearchDT.best_params_


#=======================================================================================

#build the parameter grid
LREG_grid = [ {'C':[0.5,0.9,1,1.5,2],'penalty':['l1','l2'],'solver':['liblinear']}]

#build a grid search to find the best parameters
gridsearchLREG  = GridSearchCV(model3, LREG_grid, cv=5)

#run the grid search
gridsearchLREG.fit(X_train,y_train)
best_acclg= gridsearchLREG.best_score_
best_paramlg= gridsearchLREG.best_params_

#=======================================================================================

VT.fit(X_train,y_train)

#use the VT classifier to predict
predicted=VT.predict(X_test)
predicted=list(map(int,predicted))

file=open("predicted_labels.txt","w")
for i in predicted:
    file.write(str(i)+'\n')


print(best_paramdt)
print(best_paramlg)
