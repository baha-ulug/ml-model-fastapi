import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# App creation and model loading
app = FastAPI()

def load_model():
    model = joblib.load("./outputs/model.joblib")
    return model


class PimaDiabetes(BaseModel):
    """
    Input features validation for the ML model
    """
    Pregnancies: float
    Glucose: float
    BloodPressure: float	
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post('/predict')
def predict(diabetes: PimaDiabetes):
    """
    :param PimaDiabetes: input data from the post request
    :return: predicted result
    """
    features = [[
        diabetes.Pregnancies,
        diabetes.Glucose,
        diabetes.BloodPressure,
        diabetes.SkinThickness,
        diabetes.Insulin,
        diabetes.BMI,
        diabetes.DiabetesPedigreeFunction,
        diabetes.Age,

    ]]
    model = load_model()
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)