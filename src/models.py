import io
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x

class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = SimpleAE()
    
torch.save(model.state_dict(), 'E:\\6th sem\\AI LAB\\AI_labexam\\model.pth')


from flask import Flask, request, jsonify, send_file
import torch
from torchvision import transforms
from PIL import Image

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
