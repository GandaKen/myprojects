
#Gender Classification using ; weight, height and shoe-size as features



## 1.-USING DecisionTreeClassiffier model 

from sklearn import tree
import numpy as np

#data: height, weight, shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#label
y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#classifier clf
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

#prediction given new unknown data
prediction = clf.predict([[190, 70, 43]])

#print prediction
print(" 1. Prediction  using DecisionTree is: ", prediction)


print(" ") # spacing


##. 2- USING KNeighborsClassiffier model  
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors = 1) # number neighbors =1

#build the model
knn.fit(X, y)

#make prediction. Put the data in 2D numpy array
Xpred = np.array([[190,70,43]])

#call the predic method on knn object
prediction = knn.predict(Xpred)
print(" 2. prediction using KNeighbors is: ", prediction)


print(" ") # spacing 

## . 3-  Using RandomForestClassifier model
#import the classifier
from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 10)

#Build the model
clf = clf.fit(X, y)

#Input the new value for prediction
prediction = clf.predict([[190, 70, 43]])

#print the output
print(" 3. Prediction using RandomForest is: ", prediction)

print(" ") # spacing 

##. 4 - Using Quadratic Analysis Classifier 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X =np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])

y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male'])

clf = QuadraticDiscriminantAnalysis()
QuadraticDiscriminantAnalysis(priors = None, reg_param = 0.0,store_covariance = False,
                              store_covariances = None, tol = 0.0001)

clf.fit(X, y)
print(" 4. prediction using QuadraticDiscriminantAnalysis is: ",clf.predict([[190,70,43]]))

print(" ") # spacing 

## 5 - Using NaiveBayes Classifier 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Build the model
clf.fit(X, y)
GaussianNB(priors=None, var_smoothing=1e-09)

#print the output
print(" 5. prediction using Gaussian naive Bayes is: ", clf.predict([[190,70,43]]))



