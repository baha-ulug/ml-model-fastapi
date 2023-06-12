import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def import_dataset():
    df = pd.read_csv('./localdata/diabetes.csv')
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X,y

def create_ml_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model,X_test,y_test

def dump_outputs(model, X_test, y_test):
    joblib.dump(model, "./outputs/model.joblib")
    joblib.dump(X_test, "./outputs/X_test.joblib")
    joblib.dump(y_test, "./outputs/y_test.joblib")

if __name__ == '__main__':
    X, y = import_dataset()
    model, X_test, y_test = create_ml_model(X,y)
    dump_outputs(model, X_test, y_test)
    