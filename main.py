import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = joblib.load("model/placement_model.pkl")


class StudentData(BaseModel):
    gender: str
    cgpa: float
    internships: int
    aptitude_score: int
    communication_score: int
    projects: int
    branch: str
    backlogs: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):

    form = await request.form()
    
    input_data = pd.DataFrame([{
        "gender": form["gender"],
        "cgpa": float(form["cgpa"]),
        "internships": int(form["internships"]),
        "aptitude_score": int(form["aptitude_score"]),
        "communication_score": int(form["communication_score"]),
        "projects": int(form["projects"]),
        "branch": form["branch"],
        "backlogs": int(form["backlogs"])
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "probability": round(float(probability), 3)
        }
    )
