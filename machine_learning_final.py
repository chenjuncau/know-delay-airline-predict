# https://www.kaggle.com/redhunter20/airlines-catastrophes
# https://www.kaggle.com/redhunter20/stroke-prediction-rf-acc-93

import os

#os.chdir("C:\\Users\\chenju\\Dropbox\\GT\\2021-Applied-Analytics-Practicum-CSE-6748\\final\\Data")

os.chdir("D:\\Dropbox\\GT\\2021-Applied-Analytics-Practicum-CSE-6748\\final\\Data\\LAX_Thru_IAD")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


#df = pd.read_csv('test.csv')
#df = pd.read_csv('test2.csv')
df = pd.read_csv('SFO_Final.csv')

df.head()
df.shape

df.isna().sum()

data=df
sns.pairplot(data)
plt.show()

sns.countplot(x = data['delaystatus'])


labels =data['delaystatus'].value_counts(sort = True).index
sizes = data['delaystatus'].value_counts(sort = True)
colors = ["lightblue","pink"]
plt.figure(figsize=(7,7))
plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,)
plt.title('delaystatus pie')
plt.show()



y = data['delaystatus']
#X = data[['Wind.Speed','Wind.Gust','Vis','Ceiling']]
X = data[['Temp','Dew','T.D.Spread','Wind.Direction','Altimeter','Wind.Speed','Wind.Gust','Vis','Ceiling']]

counter = Counter(y)
print(counter)

oversample = SMOTE()

X, y = oversample.fit_resample(X,y)

counter = Counter(y)
print(counter)






from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=50)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("The Training Score of RandomForestClassifier is: {:.3f}%".format(model_rf.score(X_train, y_train)*100))
print("\n----------------------------------------------------------------------\n")
print("The Classification report: \n{}\n".format(classification_report(y_test, y_pred)))

#Visualize confusion matrix
plt.figure(figsize = (10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15},
           yticklabels = ['No delay', 'delay'], xticklabels = ['Predicted no delay', 'Predicted delay'])
plt.yticks(rotation = 0)
plt.show()

print("\n----------------------------------------------------------------------\n") 
print("The Accuracy Score of RandomForestClassifier is: {:.3f}%".format(accuracy_score(y_test, y_pred)*100))






# here it is 


# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix



from imblearn.over_sampling import SMOTE

X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)



models = dict()
#models['Dicision Tree'] = DecisionTreeClassifier(class_weight={0:1,1:2})
models['Random Forest'] = RandomForestClassifier(class_weight={0:1,1:2})
models['Logistic Regression'] = LogisticRegression()
#models['Ridge Regression'] = Ridge(alpha=1.0)
models['GradientBoost'] = GradientBoostingClassifier()
models['AdaBoost'] = AdaBoostClassifier()
models['XGBoost'] = xgboost.XGBClassifier()
models['GaussianNB']= GaussianNB(var_smoothing = 1e-9)
models['KNN'] = KNeighborsClassifier(n_neighbors=3)
models['MLP'] = MLPClassifier()




for model in models:
    models[model].fit(X_train_resampled, y_train_resampled)
    print(model + ' : fit')



print("Train set prediction")
for x in models:
        
    print('------------------------'+x+'------------------------')
    model = models[x]
    y_train_pred = model.predict(X_train_resampled)
    arg_train = {'y_true':y_train_resampled, 'y_pred':y_train_pred}
    print(confusion_matrix(**arg_train))
    print(classification_report(**arg_train))



print("Test set prediction")
for x in models:
        
    print('------------------------'+x+'------------------------')
    model = models[x]
    y_test_pred = model.predict(X_test)
    arg_test = {'y_true':y_test, 'y_pred':y_test_pred}
    print(confusion_matrix(**arg_test))
    print(classification_report(**arg_test))




from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score,mean_squared_error

fig, ax = plt.subplots()
fig.set_size_inches(13,8)

for m in models:
    y_pred = models[m].predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred[:,1].ravel())
    plt.plot(fpr,tpr, label=m)
plt.xlabel('False-Positive rate')
plt.ylabel('True-Positive rate')
plt.legend()
plt.show()




print('roc_auc_score')
for i in models:
    model = models[i]
    print(i + ' : ',roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]).round(4))




from sklearn.metrics import mean_absolute_percentage_error
print('mean_absolute_percentage_error')
for i in models:
    model = models[i]
    print(i + ' : ',mean_absolute_percentage_error(y_test+1, model.predict(X_test)+1))
# plus 1 is good.

#mape = np.abs(((y_test+1) -  (models['XGBoost'].predict(X_test)+1))) / (y_test+1).mean(axis=0) * 100


# https://scikit-learn.org/stable/modules/model_evaluation.html
print('accuracy_score')
for i in models:
    model = models[i]
    print(i + ' : ',accuracy_score(y_test, model.predict(X_test)).round(4))
    
# print('RMSE')
# for i in models:
#     model = models[i]
#     print(i + ' : ',np.sqrt(mean_squared_error(y_test, model.predict(X_test))).round(4))




# # do not need to run below.

# # cross validation
# from sklearn.model_selection import cross_val_score
# #cross_val_score(model, X, y, cv=5)
# print('accuracy_score')
# for i in models:
#     model = models[i]
#     print(i + ' : ',cross_val_score(model, X_train_resampled, y_train_resampled, cv=5).mean())




# # cross validation
# from sklearn.model_selection import cross_val_score
# #cross_val_score(model, X, y, cv=5)
# print('accuracy_score')
# for i in models:
#     model = models[i]
#     print(i + ' : ',cross_val_score(model, X_train_resampled, y_train_resampled, cv=5,scoring='neg_mean_squared_error').mean())






# mean_absolute_percentage_error(y_test, models['XGBoost'].predict(X_test))


# #mean_absolute_percentage_error(models['XGBoost'].predict(X_test),y_test)

# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
# mean_absolute_percentage_error(y_true, y_pred)


#https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
