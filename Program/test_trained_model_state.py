import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
from keras import models
from screeninfo import get_monitors
import win32com.client
import mouse
import pyautogui
import time
import math

# baca video dari webcam
video = cv2.VideoCapture(0)

# baca ukuran layar laptop
screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

# load model
model = models.load_model('c:\\Users\\dafar\\Documents\\Tugas Akhir\\Program\\model_4.h5')

# load aplikasi power point
ppt_app = win32com.client.Dispatch("PowerPoint.Application")

# variabel dasar buat mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# tentukan state fungsi2 power point
navigation_state = False
pen_state = False
pointer_state = False
predict_model_state = True

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

# tipe font buat display
font = cv2.FONT_HERSHEY_SIMPLEX

# fungsi predict model
@tf.function
def predict_model(img) :
  return model(img)

# fungsi pembulatan keatas buat kalibrasi
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# program mulai baca pose tangan
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
  ) as hands:
  while True:
    _, frame = video.read()

    # ambil ukuran video webcam yang kebaca (x, y, z) => 640px x 480px 
    h, w, _ = frame.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # buat black background
    black_image = np.zeros((h, w, 3), dtype = "uint8")
    cropped_image = np.zeros((128, 128, 3), dtype = "uint8")
    flipped_image = np.zeros((128, 128, 3), dtype = "uint8")

    temp_hand_landmark = np.zeros([21,2])

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.rectangle(black_image, (128, 96), (512, 384), (0, 255, 0), 3)
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
          black_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

        for i in range(21):
          temp_hand_landmark[i, 0] = hand_landmarks.landmark[i].x # index 0 = x value
          temp_hand_landmark[i, 1] = hand_landmarks.landmark[i].y # index 1 = y value

        # ambil titik min and max dari tangan yang terdeteksi
        x_min = min(temp_hand_landmark[:, 0] * w) - 20
        y_min = min(temp_hand_landmark[:, 1] * h) - 20
        x_max = max(temp_hand_landmark[:, 0] * w) + 20
        y_max = max(temp_hand_landmark[:, 1] * h) + 20
        
        # cropped gambar
        cropped_image = black_image[int(y_min):int(y_max), int(x_min):int(x_max),:]
        # flip gambar
        flipped_image = cv2.flip(cropped_image, 1)

      #end for

      # flip display black_image
      black_image = cv2.flip(black_image, 1)

      # baca koordinat ujung teleunjuk
      index_finger_position_x = (round_up(((temp_hand_landmark[8, 0] - 0.2) * 1.6666666667), 6)*-1*screen_width)+screen_width
      index_finger_position_y = round_up(((temp_hand_landmark[8, 1] - 0.2) * 1.6666666667), 6)*screen_height

      if predict_model_state :
        #Convert the captured frame into RGB
        im = Image.fromarray(flipped_image, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = predict_model(img_array)[0]

        if prediction[0] == 1 :
          cv2.putText(black_image, 'Erase', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
          pyautogui.press('e')
        elif prediction[1] == 1 :
          cv2.putText(black_image, 'Next Slide', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
          if navigation_state == False :
            ppt_app.SlideShowWindows[0].View.Next()
            navigation_state = True
            time.sleep(1)
            navigation_state = False
        elif prediction[2] == 1 :
          cv2.putText(black_image, 'Pointer', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
          ppt_app.SlideShowWindows[0].View.LaserPointerEnabled = True
          mouse.move(index_finger_position_x, index_finger_position_y, absolute=True)
        elif prediction[3] == 1 :
          cv2.putText(black_image, 'Pen', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
          pyautogui.hotkey('ctrl', 'p')
          pen_state = True
          predict_model_state = False
        elif prediction[4] == 1 :
          cv2.putText(black_image, 'Previous Slide', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
          if navigation_state == False :
            ppt_app.SlideShowWindows[0].View.Previous()
            navigation_state = True
            time.sleep(1)
            navigation_state = False
      else :
        if pen_state :
          mouse.move(index_finger_position_x, index_finger_position_y, absolute=True)  
          gap_index_thumb_x = (temp_hand_landmark[10, 0]) - (temp_hand_landmark[4, 0])
          gap_index_thumb_y = (temp_hand_landmark[10, 1]) - (temp_hand_landmark[4, 1])

          if(abs(gap_index_thumb_x) <= 0.05 and abs(gap_index_thumb_y) <= 0.05) :
            cv2.putText(black_image, 'Pen', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)   
            mouse.press()
          else :
            mouse.release()  
    #end if
    else :
      if pen_state :
        pyautogui.press('esc')
        pen_state = False

      ppt_app.SlideShowWindows[0].View.LaserPointerEnabled = False  
      predict_model_state = True    

    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
    # since their will be most of time error of 0.001 second, we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string 
    fps = str(fps)
  
    # putting the FPS count on the frame
    cv2.putText(black_image, fps, (50, 150), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', black_image)
    key=cv2.waitKey(1)
    if key == ord('q'):
      break
  video.release()
cv2.destroyAllWindows()