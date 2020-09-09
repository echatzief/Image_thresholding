import cv2 
import numpy as np
img = None
roi = None
roi_hist = None
term_crit = None
COLOR_RANGE=20
WINDOW_NAME = "Paper Solution"

def clickCallback(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    global img,roi,roi_hist,term_crit
  
    roi = cv2.selectROI(WINDOW_NAME,img)
    roi_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    hsv_roi =  cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )  
    print(roi)

def main():
  
  # Create the window and load the video
  cam = cv2.VideoCapture("video.mp4")
  cv2.namedWindow(WINDOW_NAME)

  # Setup the callback for the click 
  cv2.setMouseCallback(WINDOW_NAME, clickCallback)

  while True:

    # Read each frame 
    ret, frame = cam.read()
    if not ret:
      print("The video is finished. Exiting...")
      break

    # Pass the frame to global variable
    global img,roi,roi_hist,term_crit
    img = frame

    if roi:
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      mask = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

      cv2.imshow("mask",mask)
      frame = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow(WINDOW_NAME,frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
      print("Escape...Exiting...")
      break
if __name__ == "__main__":
  main()