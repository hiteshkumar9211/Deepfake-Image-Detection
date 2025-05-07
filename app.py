import os
from flask import Flask, request, render_template
from model import predict_image  # Import prediction function from model.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file part"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No selected file"
            else:
                # Save and process the uploaded image
                filepath = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(filepath)
                
                # Get prediction
                result = predict_image(filepath)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)