import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model('/content/drive/MyDrive/Colab Notebooks/Dataset/nutrition.h5')

@app.route('/')
def index():
    return render_template("main.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    text=""
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        #index=['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']
        
        if pred==0:
            text="""APPLE===>
                 *Calories 95
                 *Protein  1g
                 *Carbohydrate 25g
                 *Fats 0g
                 *Dietary Fiber 4.5g
                 *Sugar 25 g
                 *Sodium 0mg
                 *Potassium 260mg"""
            print(text)
            
        elif pred==1:
            text="""BANANA===>
                 *Calories 105
                 *Protein 1.39 g
                 *carbohydrate 279g
                 *Fats 0.49g
                 *Dietary fibre 6.14g
                 *Sodium 1.2 mg
                 *Potassium 422 mg"""
            print(text)
            
        elif pred==2:
            text="""ORANGE===>
                    *Calories 105
                    *Protein 0.9g
                    *Fats 0.1g
                    *Carbohydrate 18g
                    *Dietary fiben 2.39
                    *Sugar 9g
                    *Sodium 0mg
                    *Potassium 173.8mg"""
            print(text)
            
        elif pred==3:
            text="""PINEAPPLE===>
                    *Calories 452"
                    *Portein-4.99g
                    *Fats 11g
                    *Carbohydrates -199g
                    *Dietary Fiber 139g
                    *Sugar 89g
                    *Sodium 9.1 mg
                    *Potassium 986.5mg"""
            print(text)
            
        elif pred==4:
            text="""WATERMELON===>
                    *Calories 1371
                    *Protein 26g
                    *Fats-7g
                    *Carbohydrate 341g 
                    *Dietary Fiber 18g
                    *Sugar 280g
                    *Sodium 45.2 mg
                    *Potassium  5060.2 mg"""
            print(text)
     
    return text
if __name__=='__main__':
    app.run(debug=False)