#chessbot
import cv2
import numpy as np
import time
import boardModule2
import engine
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
src = None
square_size = 100 #(pixels)
top_offset = 50 #will get rid of this
square_images = []
simple_predictions = [[None for _ in range(8)] for _ in range(8)]
previous_simple_predictions = [[None for _ in range(8)] for _ in range(8)]

from tensorflow.keras.models import load_model

#model = load_model("chess_piece_classifier.keras")
#classes = ['e', 'h', 'wp', 'wk', 'wb', 'wr', 'wq', 'wn', 'bn', 'bp', 'bk', 'bb', 'br', 'bq']

model = load_model("simple_chess_piece_classifier.keras")
classes = ['e', 'h', 'w', 'b']# empty, hand, white, black

pieces = None
cap = cv2.VideoCapture(0)

with open("data.pkl", "rb") as file:
   src = pickle.load(file)#this is why I need more comments, what is file "file" :{
if not cap.isOpened():
   print("Cannot access webcam."); exit()


def classify_square(img):
    #function that uses the AI model to identify whether an image is most likely
    #to show an empty square, a hwite piece, or a black piece
    img = cv2.resize(img, (square_size, square_size + top_offset)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img, verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx]

def get_simple_predictions(predictions): #will get rid of now piece id got rid of
    return [
        ['b' if cell.startswith('b') else
         'w' if cell.startswith('w') else
         'e' if cell in ('e', 'h') else
         cell  # fallback
         for cell in row]
        for row in predictions
    ]

def move_piece(simple_predictions, previous_simple_predictions):
    #make a list of all the places where a white piece has dissappered from (from_pos)
    # and a list of all the places wheee a white piece has appeared (to_pos)
    from_pos = []
    to_pos = []
    
    for i in range(8):
        for j in range(8):
            if previous_simple_predictions[j][i] == 'w' and simple_predictions[j][i] == 'e':
                from_pos.append(f"{chr(104 - i)}{8 - j}")
            if (previous_simple_predictions[j][i] != simple_predictions[j][i]
            and simple_predictions[j][i] == 'w'):
                to_pos.append(f"{chr(104 - i)}{8 - j}")
    
    if from_pos and to_pos: #if a move is detected
        if "e1" in from_pos: #sort out castling
            from_pos = ["e1"]
            if "c1" in to_pos:
                to_pos = ["c1"]
            elif "g1" in to_pos:
                to_pos = ["g1"]
        
        #otherwise be lazy and hope first one is right move (needs fixing/improving)
        move = from_pos[0] + to_pos[0]
        print(move)
        cv2.destroyAllWindows()
        engine.move(move)
        if from_pos[0].endswith('7') and to_pos[0].endswith('8'): #sort out promotion (auto queening)
            engine.move(move + 'q')

while True:
    ret, frame = cap.read()
    if not ret:
        break

        # Visualize detection result on warped board
    top = boardModule2.get_board(frame, src)
    key = cv2.waitKey(1) & 0xFF
    
    predictions = [[None for _ in range(8)] for _ in range(8)]
    
    for i in range(64):
        x, y = (i % 8) * square_size, top_offset + (i // 8) * square_size
        #color = (0,255,0)
        #cv2.rectangle(top, (x,y), (x+w, y+w), color, 1)
        if i < 64:
              square_img = top[y-top_offset:y+square_size, x:x+square_size]
              square_images.append(square_img)
              predictions[i//8][i%8] = classify_square(square_img)
              #cv2.imshow("Square" + str(i), square_img)
              
    #print(sum(cell == 'h' for row in predictions for cell in row))
#    for row in predictions:
 #        print(row[::-1])
  #  for row in simple_predictions:
   #      print(row[::-1])
    if sum(cell == 'h' for row in predictions for cell in row) < 2:
        predictions.reverse()
        
        simple_predictions = get_simple_predictions(predictions)
             
        
        if previous_simple_predictions is not None and previous_simple_predictions[0][0] is not None:

            move_piece(simple_predictions, previous_simple_predictions)
        
    previous_simple_predictions = engine.get_board()
#     with open("squares.pkl", "wb") as file:
#         pickle.dump(square_images, file)
      
    cv2.imshow("Board Top-Down view", top)

 #   cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('u') or key == ord('z'):
        engine.undo_move()
        previous_simple_predictions = engine.get_board()
    
    #reset the board view (if 'r' pressed (for when board or camera is moved))
    if key == ord('r'):
        src = boardModule2.get_average_src(None, cap)
        with open("data.pkl", "wb") as file:
            pickle.dump(src, file)

cap.release()
cv2.destroyAllWindows()