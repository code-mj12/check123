from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import time
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/uploads'  # Change this to the desired directory for saving result images
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    error_message = request.args.get('error_message', None)
    return render_template('index.html', error_message=error_message)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        file.close()  # Close the file
        
        main_model = YOLO('/home/omen2/Desktop/1st_iter/weights/google_best.pt')
        plant_type = main_model(filepath)
        plant_prob = plant_type[0].probs.data.tolist()
        main_confidence=np.max(plant_prob)
        plant_dict = plant_type[0].names
        selected_model = plant_dict[np.argmax(plant_prob)]
        print("The main confidence is: ",main_confidence)
        print("The selected model is: ",selected_model,type(selected_model) )
        if main_confidence < 0.9:
            # Delete the uploaded file and prompt for re-upload
            os.remove(filepath)
            print(main_confidence)
            return redirect(url_for('index', error_message="Image uploaded is not predictable. Please upload another image."))

        if selected_model == 'Rasperries':
            model = YOLO('weights/best_rasberry.pt')
        elif selected_model == 'Blackberries':
            model = YOLO('weights/best_blackberry2.pt')

        results = model(filepath)
        
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        max_prob_class = names_dict[np.argmax(probs)]
        print("class prob is",np.max(probs)," class name: ",max_prob_class)
        if (selected_model =='Rasperries' and np.max(probs) < 0.95) or (selected_model =='Blackberries' and np.max(probs) < 0.5):
            os.remove(filepath)
            print(main_confidence)
            return redirect(url_for('index', error_message="Image uploaded is not predictable. Please upload another image."))
        
        # Example processing
        img = Image.open(filepath)
        processed_img = img.copy()
        timestamp = int(time.time())  # Get current Unix timestamp
        result_image_filename = f"result_image_{timestamp}.jpg"
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], result_image_filename)
        processed_img.save(result_image_path)

        # os.remove(filepath)  # Remove the uploaded file
        c_prob= int(100*np.max(probs))
        if c_prob == 100:
            c_prob = 99
        
        return render_template('result.html', class_name=max_prob_class, probability=c_prob,
                               result_image_filename=result_image_filename,selected_model=selected_model)
    
    return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)
