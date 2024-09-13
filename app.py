from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset and define features and target variable
data = pd.read_csv("diabetes_ds.csv")
X = data.drop(columns="Outcome")
y = data["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model and save it
model_filename = "diabetes.pkl"
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, model_filename)

# Create FastAPI instance
app = FastAPI()

# Define Pydantic model for request validation
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the model outside the endpoint to avoid reloading it each time
model = joblib.load(model_filename)

# Define the prediction endpoint
@app.post("/predict")
async def predict_diabetes(input_data: DiabetesInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Interpret the prediction
    result = "diabetes" if prediction[0] == 1 else "not diabetes"
    
    return {"prediction": result}