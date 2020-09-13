import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


class LogisticModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None

    # building the model
    def build_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    # evaluating the model
    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        print("  accuracy: {0:.5f}".format(accuracy_score(self.y_test, self.y_pred)))
        print("  precision: {0:.5f}".format(precision_score(self.y_test, self.y_pred)))
        print("  recall: {0:.5f}".format(recall_score(self.y_test, self.y_pred)))
        print("  f1 score: {0:.5f}".format(f1_score(self.y_test, self.y_pred)))


def main():
    df = pd.read_csv('./titanic.csv')
    df['male'] = df['Sex'] == 'male'
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
    y = df['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    m = LogisticModel(X_train, X_test, y_train, y_test)
    m.build_model()
    m.evaluate()


if __name__ == "__main__":
    main()