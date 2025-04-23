from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from models.py import SimpleAE

app = Flask(__name__)

# Load the pretrained model
model = SimpleAE()
model.load_state_dict(torch.load('E:\\6th sem\\AI LAB\\AI_labexam\\model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'})

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert output tensor to image
    output_image = transforms.ToPILImage()(output.squeeze(0))

    # You can save the image or convert it to bytes and send it as response
    # For simplicity, let's convert it to bytes and send it
    output_image_bytes = io.BytesIO()
    output_image.save(output_image_bytes, format='JPEG')
    output_image_bytes.seek(0)

    return send_file(output_image_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
