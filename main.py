import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import prediction

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@app.get('/')
def hello_world():
  return f"Hello,World"


@app.post('/api/predict')
def predict_image(file: UploadFile = File(...)):
  print(file)
  extension = file.filename.split(".")[-1] in ("jpg", "JPEG", "jpeg", "png", "PNG", "gif")
  if not extension:
    return "Image must be jpg, png and gif format!"
  image = prediction.read_image(save_upload_file_tmp(file))
  image = prediction.preprocess(image)
  pred  = prediction.predict(image)
  print(pred)
  return pred

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
  try:
    suffix = Path(upload_file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
      shutil.copyfileobj(upload_file.file, tmp)
    tmp_path = Path(tmp.name)
  finally:
    upload_file.file.close()
  return tmp_path


class Num(BaseModel):
  num1: float
  num2: float


@app.post('/api/add')
def add(num: Num) -> float:
  sum = num.num1 + num.num2
  return sum

# if __name__ == "__main__":
#   uvicorn.run(app, host='127.0.0.1', port=8000)
