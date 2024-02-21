from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torchvision.transforms as transforms
from training.model import CNNModel2

app = Flask(__name__)

# Load the model
model = CNNModel2()
model_path = 'training/model_weights/CNNModel2_lr0.001_momentum0.99_epochs10.pth'
# training/model_weights
# list all the files in the directory
# print(os.listdir('../training/model_weights'))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the transform for input images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Ensure the uploads directory exists
uploads_dir = os.path.join(app.static_folder, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():
    # print(request.method)
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(uploads_dir, filename)
            file.save(save_path)  # Save the uploaded image
            
            image = Image.open(save_path)
            image_tensor = transform(image).unsqueeze(0)  # Preprocess the image

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)

            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_class = classes[predicted.item()]

            # Check if request is from curl or similar tool
            if 'curl' in request.headers.get('User-Agent', ''):
                return jsonify({'prediction': predicted_class})
            else:
                # Render template for browser requests
                return render_template('upload.html', prediction=predicted_class, image_name=os.path.join('uploads', filename))
    return 'Invalid request', 400


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)