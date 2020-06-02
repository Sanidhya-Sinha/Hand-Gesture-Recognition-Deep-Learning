import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/A")
    os.makedirs("data/train/B")
    os.makedirs("data/train/C")
    os.makedirs("data/train/D")
    os.makedirs("data/train/E")
    os.makedirs("data/train/F")
    os.makedirs("data/train/G")
    os.makedirs("data/train/H")
    os.makedirs("data/train/I")
    os.makedirs("data/train/J")
    os.makedirs("data/train/K")
    os.makedirs("data/train/L")
    os.makedirs("data/train/M")
    os.makedirs("data/train/N")
    os.makedirs("data/train/O")
    os.makedirs("data/train/P")
    os.makedirs("data/train/Q")
    os.makedirs("data/train/R")
    os.makedirs("data/train/S")
    os.makedirs("data/train/T")
    os.makedirs("data/train/U")
    os.makedirs("data/train/V")
    os.makedirs("data/train/W")
    os.makedirs("data/train/X")
    os.makedirs("data/train/Y")
    os.makedirs("data/train/Z")
    os.makedirs("data/test/A")
    os.makedirs("data/test/B")
    os.makedirs("data/test/C")
    os.makedirs("data/test/D")
    os.makedirs("data/test/E")
    os.makedirs("data/test/F")
    os.makedirs("data/test/G")
    os.makedirs("data/test/H")
    os.makedirs("data/test/I")
    os.makedirs("data/test/J")
    os.makedirs("data/test/K")
    os.makedirs("data/test/L")
    os.makedirs("data/test/M")
    os.makedirs("data/test/N")
    os.makedirs("data/test/O")
    os.makedirs("data/test/P")
    os.makedirs("data/test/Q")
    os.makedirs("data/test/R")
    os.makedirs("data/test/S")
    os.makedirs("data/test/T")
    os.makedirs("data/test/U")
    os.makedirs("data/test/V")
    os.makedirs("data/test/W")
    os.makedirs("data/test/X")
    os.makedirs("data/test/Y")
    os.makedirs("data/test/Z")


# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    temp = cv2.imread('image.png')
    cv2.imshow('template',temp)
    # Getting count of existing images
    count = {'A': len(os.listdir(directory+"/A")),
             'B': len(os.listdir(directory+"/B")),
             'C': len(os.listdir(directory+"/C")),
             'D': len(os.listdir(directory+"/D")),
             'E': len(os.listdir(directory+"/E")),
             'F': len(os.listdir(directory+"/F")),
             'G': len(os.listdir(directory+"/G")),
             'H': len(os.listdir(directory+"/H")),
             'I': len(os.listdir(directory+"/I")),
             'J': len(os.listdir(directory+"/J")),
             'K': len(os.listdir(directory+"/K")),
             'L': len(os.listdir(directory+"/L")),
             'M': len(os.listdir(directory+"/M")),
             'N': len(os.listdir(directory+"/N")),
             'O': len(os.listdir(directory+"/O")),
             'P': len(os.listdir(directory+"/P")),
             'Q': len(os.listdir(directory+"/Q")),
             'R': len(os.listdir(directory+"/R")),
             'S': len(os.listdir(directory+"/S")),
             'T': len(os.listdir(directory+"/T")),
             'U': len(os.listdir(directory+"/U")),
             'V': len(os.listdir(directory+"/V")),
             'W': len(os.listdir(directory+"/W")),
             'X': len(os.listdir(directory+"/X")),
             'Y': len(os.listdir(directory+"/Y")),
             'Z': len(os.listdir(directory+"/Z"))}
             
            
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 1)
    #cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "ONE : "+str(count['one']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "TWO : "+str(count['two']), (10, 105), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "THREE : "+str(count['three']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "FOUR : "+str(count['four']), (10, 135), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "FIVE : "+str(count['five']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "A : "+str(count['A']), (10, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "B : "+str(count['B']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "C : "+str(count['C']), (10, 105), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "D : "+str(count['D']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "E : "+str(count['E']), (10, 135), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "F : "+str(count['F']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "G : "+str(count['G']), (10, 165), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "H : "+str(count['H']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "I : "+str(count['I']), (10, 195), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "J : "+str(count['J']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "K : "+str(count['K']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "L : "+str(count['L']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "M : "+str(count['M']), (10, 255), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "N : "+str(count['N']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "O : "+str(count['O']), (10, 285), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "P : "+str(count['P']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "Q : "+str(count['Q']), (10, 315), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "R : "+str(count['R']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "S : "+str(count['S']), (10, 345), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "T : "+str(count['T']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "U : "+str(count['U']), (10, 375), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "V : "+str(count['V']), (10, 390), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "W : "+str(count['W']), (10, 405), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "X : "+str(count['X']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "Y : "+str(count['Y']), (10, 435), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    cv2.putText(frame, "Z : "+str(count['Z']), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('A'):
        cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', roi)
    if interrupt & 0xFF == ord('B'):
        cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', roi)
    if interrupt & 0xFF == ord('C'):
        cv2.imwrite(directory+'C/'+str(count['C'])+'.jpg', roi)
    if interrupt & 0xFF == ord('D'):
        cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', roi)
    if interrupt & 0xFF == ord('E'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', roi)
    if interrupt & 0xFF == ord('F'):
        cv2.imwrite(directory+'F/'+str(count['F'])+'.jpg', roi)
    if interrupt & 0xFF == ord('G'):
        cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', roi)
    if interrupt & 0xFF == ord('H'):
        cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', roi)
    if interrupt & 0xFF == ord('I'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', roi)
    if interrupt & 0xFF == ord('J'):
        cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', roi)
    if interrupt & 0xFF == ord('K'):
        cv2.imwrite(directory+'K/'+str(count['K'])+'.jpg', roi)
    if interrupt & 0xFF == ord('L'):
        cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', roi)
    if interrupt & 0xFF == ord('M'):
        cv2.imwrite(directory+'M/'+str(count['M'])+'.jpg', roi)
    if interrupt & 0xFF == ord('N'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', roi)
    if interrupt & 0xFF == ord('O'):
        cv2.imwrite(directory+'O/'+str(count['O'])+'.jpg', roi)
    if interrupt & 0xFF == ord('P'):
        cv2.imwrite(directory+'P/'+str(count['P'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Q'):
        cv2.imwrite(directory+'Q/'+str(count['Q'])+'.jpg', roi)
    if interrupt & 0xFF == ord('R'):
        cv2.imwrite(directory+'R/'+str(count['R'])+'.jpg', roi)
    if interrupt & 0xFF == ord('S'):
        cv2.imwrite(directory+'S/'+str(count['S'])+'.jpg', roi)
    if interrupt & 0xFF == ord('T'):
        cv2.imwrite(directory+'T/'+str(count['T'])+'.jpg', roi)
    if interrupt & 0xFF == ord('U'):
        cv2.imwrite(directory+'U/'+str(count['U'])+'.jpg', roi)
    if interrupt & 0xFF == ord('V'):
        cv2.imwrite(directory+'V/'+str(count['V'])+'.jpg', roi)
    if interrupt & 0xFF == ord('W'):
        cv2.imwrite(directory+'W/'+str(count['W'])+'.jpg', roi)
    if interrupt & 0xFF == ord('X'):
        cv2.imwrite(directory+'X/'+str(count['X'])+'.jpg', roi) 
    if interrupt & 0xFF == ord('Y'):
        cv2.imwrite(directory+'Y/'+str(count['Y'])+'.jpg', roi)
    if interrupt & 0xFF == ord('Z'):
        cv2.imwrite(directory+'Z/'+str(count['Z'])+'.jpg', roi)
        
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""
