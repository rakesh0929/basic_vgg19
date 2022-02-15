
import numpy as np
import os
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.vgg19 import preprocess_input
# from flask_ngrok import run_with_ngrok

from flask import Flask,redirect,url_for,render_template,request

app = Flask(__name__)
# run_with_ngrok(app)
model_path = "/content/model_vgg.h5"

model = load_model(model_path)
model.make_predict_function()

def model_predict(img_path,model):
  img = image.load_img(img_path,target_size=(224,224))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds

@app.route("/",methods=["GET"])
def index():
  return render_template("/content/drive/MyDrive/project_vgg19/templates/index.html")
@app.route("/predict",methods=["GET","POST"])
def upload():
  if request.method == "POST":
    f = request.files["file"]
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, '/content/drive/MyDrive/project_vgg19/upload', secure_filename(f.filename))
    f.save(file_path)

    preds = model_predict(file_path,model)

    pred_class = decode_predictions(preds,top=1)
    result = str(pred_class[0][0][1])
    return result
  return None

if __name__ == "__main__":
  app.run(debug=True)