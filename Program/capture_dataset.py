import cv2
import mediapipe as mp
# import pandas as pd
import numpy as np
import os
import datetime
import time
# import xlsxwriter

def GetFileName():
  x = datetime.datetime.now()
  s = x.strftime('%Y-%m-%d-%H%M%S%f')
  return s

def CreateDir(path):
  ls = [];
  head_tail = os.path.split(path)
  ls.append(path)
  while len(head_tail[1])>0:
    head_tail = os.path.split(path)
    path = head_tail[0]
    ls.append(path)
    head_tail = os.path.split(path)   
  for i in range(len(ls)-2,-1,-1):
    sf =ls[i]
    isExist = os.path.exists(sf)
    if not isExist:
      os.makedirs(sf)

NamaDataSet = "pointer"
# DirektoriData = "c:\\temp\\dataimage"+"\\"+NamaDataSet+"\\"+GetFileName()
DirektoriData = "c:\\Users\\dafar\\Documents\\Project TA\\base_data\\testing"+"\\"+NamaDataSet
CreateDir(DirektoriData)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

TimeStart = time.time() 
TimeNow = time.time() +10
FrameRate = 5

# workbook = xlsxwriter.Workbook('next_landmark_coordinate.xlsx')
# worksheet = workbook.add_worksheet()
frame = 0
row = 0
col = 0

# for x in range(21) :  
#   worksheet.write(0, x+2, x+1)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    # get image size (x, y, z)
    h, w, _ = image.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # create black background
    black_image = np.zeros((h, w, 3), dtype = "uint8")

    list_hand_landmarks = []

    # create x y min max
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
          black_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

        # Append hand landmarks (x, y, z) to  list_hand_landmarks
        temp_hand_landmark = np.zeros([21,3])
        for i in range(21):
          temp_hand_landmark[i, 0] = hand_landmarks.landmark[i].x * w # index 0 = x value
          temp_hand_landmark[i, 1] = hand_landmarks.landmark[i].y * h # index 1 = y value
          temp_hand_landmark[i, 2] = hand_landmarks.landmark[i].z  # index 1 = y value
        
        list_hand_landmarks.append(temp_hand_landmark)

        # Get min and max coordinate by each hand
        x_min = min(temp_hand_landmark[:, 0]) - 64
        y_min = min(temp_hand_landmark[:, 1]) - 64
        x_max = max(temp_hand_landmark[:, 0]) + 64
        y_max = max(temp_hand_landmark[:, 1]) + 64
        
        # Draw rectangle for hand landmark
        # cv2.rectangle(black_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)

        cropped_image = black_image[int(y_min):int(y_max), int(x_min):int(x_max),:]
        flipped_image = cv2.flip(cropped_image, 1)
        resized_image = cv2.resize(flipped_image, (128,128))
        # cropped_image = black_image[int(y_min) - 10:int(y_min) + 310, int(x_min) - 10:int(x_min) + 310,:]
        
        # dy =int(y_max) - int(y_min)
        # dx = int(x_max) - int(x_min)
        # print(dy,dx)
        # print(cropped_image.shape)

        # Save frame
        TimeNow = time.time() 
        if TimeNow-TimeStart>1/FrameRate:
          # row += 1
          # if row%2 != 0 :
          #   frame += 1
          #   worksheet.write(row, 0, "frame " + str(frame))
          #   worksheet.write(row, 1, "X")
          #   worksheet.write(row+1, 1, "Y")
          # for i in range(21) : 
          #   worksheet.write(row, i+2, temp_hand_landmark[i, 0])
          #   worksheet.write(row+1, i+2, temp_hand_landmark[i, 0])
            
          # df = pd.DataFrame(temp_hand_landmark).T
          # df.to_excel(excel_writer = "C:/Users/dafar/Documents/Project TA/landmark_coordinate.xlsx")
          frame += 1
          TimeStart = TimeNow
          sFile = DirektoriData+"\\"+str(frame) + "_" + NamaDataSet
          cv2.imwrite(sFile+'.jpg', resized_image)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(black_image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
# workbook.close()
cap.release()