import tensorflow as tf
import joblib
with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()


loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model.load_weights("model_weights.weights.h5")

with open("model_architecture1.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model1 = tf.keras.models.model_from_json(loaded_model_json)
loaded_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model1.load_weights("model_weights.weights1.h5")


mlp=joblib.load('mlp.sav')
mlp1=joblib.load('mlp1.sav')
resnet=joblib.load('resnet.sav')
resnet1=joblib.load('resnet1.sav')


from flask import Flask, render_template, request, url_for,send_from_directory,Response
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
import os



app = Flask(__name__)
app.config["SECRET_KEY"] = 'ajashjkjm'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



list1=["no cancer","cancer"]
list2=["actinic keratosis",'basal cell carcinoma','dermatofibroma','melanoma','nevus','pigmented benign keratosis','seborrheic keratosis','squamous cell carcinoma','vascular lesion']

@app.route('/')
def home():
    return render_template("main.html")

@app.route('/cells')
def cells():
    return render_template("cells.html")
def re_size(filepath,loaded_model):
    img = image.load_img(filepath,target_size=(250,250))
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values between 0 and 1
    val1 = loaded_model.predict(img_array)  # Assuming loaded_model is defined elsewhere
    val1 = np.argmax(val1)
    return val1


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/file', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        model=request.form["name"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            img_path="uploads\\"+filename
            if(model=='mlp'):
                prediction=re_size(img_path,mlp)
            elif(model=='resnet'):
                prediction=re_size(img_path,resnet)
            else:
                prediction=re_size(img_path,loaded_model)
            return render_template('file.html', prediction=list1[prediction],file_url=file_url)

    return render_template('file.html', prediction=None,file_url=None)










@app.route('/cell_prediction', methods=['GET', 'POST'])
def start_page():
    if request.method == 'POST':
        file = request.files['file']
        model = request.form["name"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            img_path="uploads\\"+filename
            if (model == 'mlp'):
                prediction = re_size(img_path, mlp1)
            elif (model == 'resnet'):
                prediction = re_size(img_path, resnet1)
            else:
                prediction = re_size(img_path, loaded_model1)
            return render_template('cells.html', prediction=list2[prediction],file_url=file_url)

    return render_template('cells.html', prediction=None,file_url=None)







if __name__ == '__main__':
    app.run(debug=True)

