import fastapi
import uvicorn

from joblib import load

from pydantic import BaseModel
from nepalitokenizer import NepaliTokenizer
from sklearn.linear_model import LogisticRegression


app = fastapi.FastAPI()



class NepaliText(BaseModel):
    pramas: str




class NepaliTextClassification:

    def __init__(self):
        self.clf: LogisticRegression = load("clf.bin")
        self.tokenize = NepaliTokenizer()


    def predict(self, item: NepaliText):
        self.text = self.tokenize.tokenizer(item.pramas)
        label = self.clf.predict(self.text)[0]
        return label



@app.get("/")
def root():
    return {"GoTo": "/docs"}



@app.post("/")
def get_label(item: NepaliText):
    classification = NepaliTextClassification()
    classify = classification.predict(item)
    return {"label": classify}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
