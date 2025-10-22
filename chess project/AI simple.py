#class with the AI for the piece detection
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configure
image_width = 100
image_height = 150

data_dir = "labeled_pieces"
classes = ['e', 'h', 'w', 'b']
label_map = {cls: idx for idx, cls in enumerate(classes)}

# Load images and labels
X = []
y = []

for filename in os.listdir(data_dir):
    for cls in classes:
        #if cls in filename:
        if (f"{cls}." in filename and (cls == 'e' or cls == 'h')) or (filename[-6] == cls):
            img = cv2.imread(os.path.join(data_dir, filename))
            img = cv2.resize(img, (image_width, image_height))
           # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img)
            y.append(label_map[cls])
            break

print(len(X))
X = np.array(X) / 255.0  # Normalize
y = to_categorical(y, num_classes=len(classes))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 100, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False  # speeds up training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test))
model.save("simple_chess_piece_classifier.keras")

