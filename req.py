import requests
import joblib


def select_random_rows(X, y,):
    random_indices = X.sample(n=1, random_state=42).index
    X_selected = X.loc[random_indices]
    y_selected = y.loc[random_indices]
    return X_selected, y_selected

def make_req(x,y):
    row = x.iloc[0]
    features = {
            "Pregnancies": row["Pregnancies"],
            "Glucose": row["Glucose"],
            "BloodPressure": row["BloodPressure"],	
            "SkinThickness": row["SkinThickness"],
            "Insulin": row["Insulin"],
            "BMI": row["BMI"],
            "DiabetesPedigreeFunction": row["DiabetesPedigreeFunction"],
            "Age": row["Age"]
        }
    resp = requests.post( "http://127.0.0.1:80/predict", json=features)
    print(f"Input data: {features}")
    print(f"Predicted Value: {resp.json().get('prediction')}")
    print(f"Actural Value: {y.iloc[0]}")

if __name__ == '__main__':
    X_test = joblib.load("./outputs/X_test.joblib")
    y_test = joblib.load("./outputs/y_test.joblib")
    X_selected, y_selected = select_random_rows(X_test, y_test)
    make_req(X_selected, y_selected)