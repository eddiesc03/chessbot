import cv2
import os
import pickle

# Creates a folder to store labeled images
output_dir = "training_pics"
os.makedirs(output_dir, exist_ok=True)

# List of piece square images that want labelling
with open("squares.pkl", "rb") as file:
   square_images = pickle.load(file)

# Dictionary to store labels
# Try to load existing labels
labels = {}
if os.path.exists("piece_labels.pkl"):
    with open("piece_labels.pkl", "rb") as f:
        labels = pickle.load(f)

n = 0

for i, img in enumerate(square_images):

    if key == 27:  # ESC to quit
        break

    if n < 16: label = 'w'
    elif n < 32: label = 'e'
    elif n < 48: label = 'e'
    elif n < 64: label = 'b'
    n += 1
    
    if label:
        # Generate a unique filename (e.g. continue from existing count)
        existing_files = os.listdir(output_dir)
        next_index = len(existing_files)

        filename = f"{next_index}_{label}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        print(filepath)
        labels[filename] = label
    else:
        print("Invalid key, skipping this image.")

# Save labels as pickle for future use
with open("piece_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Labeling complete. Dataset saved.")