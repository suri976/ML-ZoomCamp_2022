import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class PatientInput(BaseModel):
    age: float
    anaemia: str
    creatinine_phosphokinase: int
    diabetes: str
    ejection_fraction: int
    high_blood_pressure: str
    platelets: float
    serum_creatinine: float
    serum_sodium: int
    sex: str
    smoking: str
    time: int

    
model_ref = bentoml.xgboost.get("heart_failure_xgboost:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()


svc = bentoml.Service("heart_failure_classifier", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=PatientInput), output=JSON())
async def classify(input_patient):
    """
    Placing prediction service on API point that functions as a receiver
    to JSON input and returns a prediction output.
    This only works for one client. 
    """
    application_data = input_patient.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)

    result = prediction[0]  

    if result > 0.5:
        return {"Patient condition status": "Deceased"}
    else:
        return {"Patient condition status": "Survived"}

