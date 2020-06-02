import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
#categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'A', 7: 'B', 8: 'C',
#              9: 'D', 10: 'E', 11: 'F', 12: 'G', 13: 'H', 14: 'I', 15: 'J', 16: 'K', 17: 'L', 18: 'M',
#              19: 'N', 20: 'O', 21: 'P', 22: 'Q', 23: 'R', 24: 'S', 25: 'T',26: 'U', 27: 'V', 28: 'W',
#              29: 'X', 30: 'Y', 31: 'Z'}

categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
              10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
              19: 'T',20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:
    
    temp = cv2.imread('image.png')
    cv2.imshow('template',temp)
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    '''  prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'A': result[0][6], 
                  'B': result[0][7], 
                  'C': result[0][8],
                  'D': result[0][9],
                  'E': result[0][10],
                  'F': result[0][11],
                  'G': result[0][12], 
                  'H': result[0][13], 
                  'I': result[0][14],
                  'J': result[0][15],
                  'K': result[0][16],
                  'L': result[0][17],
                  'M': result[0][18], 
                  'N': result[0][19], 
                  'O': result[0][20],
                  'P': result[0][21],
                  'Q': result[0][22],
                  'R': result[0][23],
                  'S': result[0][24], 
                  'T': result[0][25], 
                  'U': result[0][26],
                  'V': result[0][27],
                  'W': result[0][28],
                  'X': result[0][29],
                  'Y': result[0][30],
                  'Z': result[0][31]} '''
    
    prediction = {'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6], 
                  'H': result[0][7], 
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12], 
                  'N': result[0][13], 
                  'O': result[0][14],
                  'P': result[0][15],
                  'Q': result[0][16],
                  'R': result[0][17],
                  'S': result[0][18], 
                  'T': result[0][19], 
                  'U': result[0][20],
                  'V': result[0][21],
                  'W': result[0][22],
                  'X': result[0][23],
                  'Y': result[0][24],
                  'Z': result[0][25]}
    
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    #cv2.putText(frame, prediction[0][0], (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,255), 6)    
    cv2.putText(frame, prediction[0][0], (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,255,255), 3) 
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
