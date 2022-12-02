from flask import Flask,render_template,request
import os
import numpy as np
from keras.models import load_model
from keras_preprocessing import image
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__,template_folder="templates")

model = load_model('nutrition.h5')

print('loaded model from disk')

app.config['IMAGE_UPLOADS'] = "uploads/"
@app.route('/')
def home():
    return render_template("Home.html")
@app.route('/image' ,methods=['Get','Post'])
def image1():
    return render_template("image.html")
@app.route('/manual' ,methods=['Get','Post'])
def manual():
    return render_template("manual.html")

@app.route('/predict',methods=['Get','Post'])
def launch():
    f = request.files['file']
    filename = secure_filename(f.filename)
    basedir =  os.path.abspath(os.path.dirname(__file__))
    f.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
    p = "uploads/"+filename
    img = image.load_img(p,grayscale=False,target_size=(64,64))
    x= image.img_to_array(img)
    x =np.expand_dims(x,axis= 0)
    pred =model.predict(x)
    pred = pred.astype('int32')
    n = np.array(pred[0])
    s = np.where(n==1)
    index= ['APPLE','BANANA','ORANGE','PINEAPPLE','WATERMELON'] 
    n=int(s[0])
    result=(index[n])
    apiResult=nutrition(result)
    final_result = {
            "result" : result, 
            "apiResult" : apiResult
        }
    return final_result

def nutrition(index):
    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"
    querystring = {"query":index}
    headers = {
	"X-RapidAPI-Key": "7c2fb6a502msh4e99d771797d074p173659jsnf288c18cf37c",
	"X-RapidAPI-Host": "calorieninjas.p.rapidapi.com"
    }
    # response = requests.request("GET", url, headers=headers, params=querystring)
    # return response.text
    
    response = requests.request("GET", url="https://calorieninjas.p.rapidapi.com/v1/nutrition", headers =
                                {
        'x-rapidapi-key': "5d797ab107mshe668f26bd044e64p1ffd34jsnf47bfa9a8ee4",
        'x-rapidapi-host': "calorieninjas.p.rapidapi.com"
        }, params= {"query":index} )
    print(response.text)     
    return response.json()['items']
    
    
    

if __name__== "__main__":
    app.run(debug=False)