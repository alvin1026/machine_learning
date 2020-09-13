import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np

df = pd.read_csv('./titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3, shuffle=True)
kf_split = kf.split(X)
print(kf_split)
kf_split_list = list(kf_split)
print(type(kf_split))
print(type(kf_split_list[0][1][1]))
print(list(kf.split(X)))

for train, test in kf.split(X):
    print(train, test)

