#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import optuna
from optuna import Trial, visualization
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('../input/cascade-cup-22/train.csv')


train['order_hour'] = pd.to_datetime(train['order_time']).dt.hour
train['order_minute'] = pd.to_datetime(train['order_time']).dt.minute

train['allot_hour1'] = pd.to_datetime(train['allot_time']).dt.hour
train['allot_minute1'] = pd.to_datetime(train['allot_time']).dt.minute

train['accept_hour2'] = pd.to_datetime(train['accept_time']).dt.hour
train['accept_minute2'] = pd.to_datetime(train['accept_time']).dt.minute


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.distplot(train['accept_hour2'])


sns.distplot(train['order_minute'])


sns.distplot(train['last_mile_distance'])


corr = train.corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
plt.yticks(rotation=0)
plt.show()

test = pd.read_csv('../input/cascade-cup-22/test.csv')


test['order_hour'] = pd.to_datetime(test['order_time']).dt.hour
test['order_minute'] = pd.to_datetime(test['order_time']).dt.minute

test['allot_hour1'] = pd.to_datetime(test['allot_time']).dt.hour
test['allot_minute1'] = pd.to_datetime(test['allot_time']).dt.minute

test['accept_hour2'] = pd.to_datetime(test['accept_time']).dt.hour
test['accept_minute2'] = pd.to_datetime(test['accept_time']).dt.minute
train['reassignment_reason'].unique()
train['order_time'] = pd.to_datetime(train['order_time'])
train['accept_time'] = pd.to_datetime(train['accept_time'])
train['allot_time'] = pd.to_datetime(train['allot_time'])
train['Acceptance_Time'] = (train['accept_time'] - train['allot_time'])
train['Allotment_Time'] = (train['allot_time'] - train['order_time'])

test['order_time'] = pd.to_datetime(test['order_time'])
test['accept_time'] = pd.to_datetime(test['accept_time'])
test['allot_time'] = pd.to_datetime(test['allot_time'])
test['Acceptance_Time'] = (test['accept_time'] - test['allot_time'])
test['Allotment_Time'] = (test['allot_time'] - test['order_time'])

train['Acceptance_Time'] = pd.to_numeric(train['Acceptance_Time'].dt.seconds,downcast = 'integer')
test['Acceptance_Time'] = pd.to_numeric(test['Acceptance_Time'].dt.seconds,downcast = 'integer')

train['Acceptance_Time']=train['Acceptance_Time'].fillna(0)
test['Acceptance_Time']=test['Acceptance_Time'].fillna(0)

train['Allotment_Time'] = pd.to_numeric(train['Allotment_Time'].dt.seconds,downcast = 'integer')
test['Allotment_Time'] = pd.to_numeric(test['Allotment_Time'].dt.seconds,downcast = 'integer')

train['Allotment_Time']=train['Allotment_Time'].fillna(0)
test['Allotment_Time']=test['Allotment_Time'].fillna(0)

train['session_time']=train['session_time'].fillna(180)
test['session_time']=test['session_time'].fillna(180)

train.drop(['allot_time','accept_time','pickup_time','delivered_time','order_id'],axis =1,inplace= True)
test.drop(['allot_time','accept_time','order_id'],axis =1,inplace= True)

train['reassigned_order']=train['reassigned_order'].fillna(0)
test['reassigned_order']=test['reassigned_order'].fillna(0)

train['reassignment_reason']=train['reassignment_reason'].fillna('0')
test['reassignment_reason']=test['reassignment_reason'].fillna('0')

train['reassignment_method']=train['reassignment_method'].fillna('0')
test['reassignment_method']=test['reassignment_method'].fillna('0')

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
train["reassigned_order"] = ord_enc.fit_transform(train[["reassigned_order"]])
test["reassigned_order"] = ord_enc.fit_transform(test[["reassigned_order"]])

train["reassignment_reason"] = ord_enc.fit_transform(train[["reassignment_reason"]])
test["reassignment_reason"] = ord_enc.fit_transform(test[["reassignment_reason"]])

train['reassignment_method'] = ord_enc.fit_transform(train[["reassignment_method"]])
test["reassignment_method"] = ord_enc.fit_transform(test[["reassignment_method"]])

train['order_time'] = pd.to_datetime(train['order_time']).dt.time
train['order_date'] = pd.to_datetime(train['order_date']).dt.date

test['order_time'] = pd.to_datetime(test['order_time']).dt.time
test['order_date'] = pd.to_datetime(test['order_date']).dt.date
train = train.dropna(axis=0,subset =['undelivered_orders'])
train['efficiency'] = train['delivered_orders'] / train['alloted_orders']
test['efficiency'] = test['delivered_orders'] / test['alloted_orders']
train['total_distance']= train['first_mile_distance'] + train['last_mile_distance']
test['total_distance']= test['first_mile_distance'] + test['last_mile_distance']
train.drop(['alloted_orders','undelivered_orders','first_mile_distance','total_distance'],axis =1,inplace= True)
test.drop(['alloted_orders','undelivered_orders','first_mile_distance','total_distance'],axis =1,inplace= True)
train.drop(['order_time','reassignment_method','reassigned_order','reassignment_reason'],axis =1,inplace= True)
test.drop(['order_time','reassignment_method','reassigned_order','reassignment_reason'],axis =1,inplace= True)
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
train["order_date"] = ord_enc.fit_transform(train[["order_date"]])
test["order_date"] = ord_enc.fit_transform(test[["order_date"]])
test['lifetime_order_count'] = test['lifetime_order_count'].fillna(828)
test['efficiency'] = test['efficiency'].fillna(0.99)
canctime=train[~train['cancelled_time'].isnull()]
nocanctime=train[train['cancelled_time'].isnull()]

nocanctime.drop(['cancelled_time'],axis =1,inplace= True)
canctime.drop(['cancelled_time'],axis =1,inplace= True)
train.drop(['cancelled_time'],axis =1,inplace= True)

feature_cols = [col for col in train.columns if col not in ['cancelled']]
target_col = ['cancelled']

corr = train.corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(corr, annot=True, linewidths=.5, ax=ax)
plt.yticks(rotation=0)
plt.show()
train['accept_hour2']=train['accept_hour2'].fillna(13)
train['accept_minute2']=train['accept_minute2'].fillna(30)
X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train[target_col], 
                                                    stratify=train[target_col], 
                                                    test_size=0.1, random_state=42, 
                                                    shuffle=True)

test['delivered_orders']=test['delivered_orders'].fillna(108)
test['accept_hour2']=test['accept_hour2'].fillna(13)
test['accept_minute2']=test['accept_minute2'].fillna(30)


model1 = XGBClassifier( n_estimators=600, learning_rate=0.013185216741789522,
                       subsample=0.9, colsample_bytree=0.8,
                       reg_alpha=6, reg_lambda=5, max_depth=5, 
                       min_child_weight=184, scale_pos_weight=45)
model1.fit(X_train, y_train, eval_metric='auc')

print(f"Training ROC: {roc_auc_score(y_train, model1.predict_proba(X_train)[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, model1.predict_proba(X_test)[:, 1])}")

model2 = CatBoostClassifier()
model2.fit(X_train, y_train)

print(f"Training ROC: {roc_auc_score(y_train, model2.predict_proba(X_train)[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1])}")

model3 = LGBMClassifier()
model3.fit(X_train, y_train)

print(f"Training ROC: {roc_auc_score(y_train, model3.predict_proba(X_train)[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, model3.predict_proba(X_test)[:, 1])}")

model4 =GradientBoostingClassifier()
model4.fit(X_train, y_train)

print(f"Training ROC: {roc_auc_score(y_train, model4.predict_proba(X_train)[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, model4.predict_proba(X_test)[:, 1])}")

model5 =RandomForestClassifier()
model5.fit(X_train, y_train)

print(f"Training ROC: {roc_auc_score(y_train, model5.predict_proba(X_train)[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, model5.predict_proba(X_test)[:, 1])}")


from mlxtend.classifier import StackingCVClassifier
stack= StackingCVClassifier(classifiers=(model1,model2,model3,model4,model5),
                                meta_classifier=model1,
                                use_features_in_secondary=True)

stack_= stack.fit(np.array(X_train), np.array(y_train))

print(f"Training ROC: {roc_auc_score(y_train, stack_.predict_proba(np.array(X_train))[:, 1])}")
print(f"Validation ROC: {roc_auc_score(y_test, stack_.predict_proba(np.array(X_test))[:, 1])}")

submission = pd.read_csv('../input/cascade-cup-22/sample_submission.csv')

preds1 = stack_.predict_proba(np.array(test))[:, 1]

submission['cancelled'] = preds1
submission.to_csv('submission.csv', index=False)

submission
