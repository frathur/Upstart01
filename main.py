import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Define the data model for incoming loan applications
class LoanApplication(BaseModel): 
    no_of_dependents: int
    education: int         # 1 for Graduate, 0 for Not Graduate
    self_employed: int     # 1 for Yes, 0 for No
    income_annum: float    # Normalized value
    loan_amount: float     # Normalized value
    loan_term: float       # Normalized value
    cibil_score: float     # Normalized value
    residential_assets_value: float  # Normalized value
    commercial_assets_value: float   # Normalized value
    luxury_assets_value: float       # Normalized value
    bank_asset_value: float          # Normalized value

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS middleware to allow requests from any origin (for development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model when the API starts
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error loading the model:", e)
    model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Loan Approval Prediction API!"}

@app.post("/predict")
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # Convert input data into the correct format for prediction
        data = np.array([[ 
            application.no_of_dependents,
            application.education,
            application.self_employed,
            application.income_annum,
            application.loan_amount,
            application.loan_term,
            application.cibil_score,
            application.residential_assets_value,
            application.commercial_assets_value,
            application.luxury_assets_value,
            application.bank_asset_value
        ]])
        
        # Perform prediction
        prediction = model.predict(data)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        return {"loan_status": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
