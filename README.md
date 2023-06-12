# ml-model-fastapi
# Project Description

This project is an AI project consisting of three Python files: `app.py`, `train.py`, and `req.py`. In this project, a diabetes prediction model is trained and predictions are served to users through a FastAPI application.

## File Descriptions

- `app.py`: The main file that starts the FastAPI application and provides predictions using the trained model.
- `train.py`: Imports the training dataset, creates a machine learning model, and saves the model and test dataset to output files.
- `req.py`: An example Python script used to send HTTP requests to the application.

## Installation and Usage

1. Clone the project:

```bash
git clone https://github.com/user/project.git
cd project
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Prepare the training dataset: Place the diabetes dataset into
```bash
 ./localdata/diabetes.csv
```
4. Train the machine learning model:
```bash
python train.py
```
5. Start the application:
```bash
python app.py
```
You can use the req.py file to send HTTP requests to the application.

## API Endpoints
- POST /predict: A POST request used to make a diabetes prediction. The request body should contain the required features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age). The predicted result is returned.