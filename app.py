# =========================================================
# DRIVER DROWSINESS DETECTION STREAMLIT APP
# =========================================================

import streamlit as st

import torch
import torch.nn as nn

import torchvision.transforms as transforms

from PIL import Image

# =========================================================
# PAGE TITLE
# =========================================================

st.title('Driver Drowsiness Detection')

st.write('Eye Closure and Yawning Detection using CNN')

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform_value = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((28,28)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])

# =========================================================
# CNN MODEL
# =========================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(

            in_channels=1,

            out_channels=32,

            kernel_size=3,

            padding=1

        )

        self.pool = nn.MaxPool2d(

            kernel_size=2,

            stride=2

        )

        self.conv2 = nn.Conv2d(

            in_channels=32,

            out_channels=64,

            kernel_size=3,

            padding=1

        )

        self.fc1 = nn.Linear(

            7 * 7 * 64,

            128

        )

        self.fc2 = nn.Linear(

            128,

            4

        )

    def forward(self, x):

        x = self.pool(

            torch.relu(

                self.conv1(x)

            )

        )

        x = self.pool(

            torch.relu(

                self.conv2(x)

            )

        )

        x = x.view(-1, 7 * 7 * 64)

        x = torch.relu(

            self.fc1(x)

        )

        x = self.fc2(x)

        return x

# =========================================================
# LOAD MODEL
# =========================================================

model = CNN()

model.load_state_dict(

    torch.load(

        'models/drowsiness_cnn_model.pth',

        map_location=torch.device('cpu')

    )

)

model.eval()

# =========================================================
# CLASS NAMES
# =========================================================

classes = [

    'Closed',

    'Open',

    'no_yawn',

    'yawn'

]

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    'Upload Driver Image',

    type=['jpg','jpeg','png']

)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', width=300)

    image_tensor = transform_value(image)

    image_tensor = image_tensor.unsqueeze(0)

    output = model(image_tensor)

    _, prediction = torch.max(output,1)

    predicted_class = classes[prediction.item()]

    st.subheader(f'Predicted Class: {predicted_class}')

    # =====================================================
    # FATIGUE LOGIC
    # =====================================================

    if predicted_class in ['Open','no_yawn']:

        fatigue = 'ALERT'

    elif predicted_class == 'yawn':

        fatigue = 'MILD FATIGUE'

    else:

        fatigue = 'SEVERE FATIGUE'

    st.success(f'Driver Status: {fatigue}')