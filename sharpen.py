import cv2
import numpy as np
img = None

COLOR_RANGE=0.30
WINDOW_NAME = "Video"

lowers = []
uppers = []
pixels = []

# Retrieve the image pixel and define the lower and the upper
def capture_click(event,x,y,flags,param):
    global img,lowers,uppers
    if event == cv2.EVENT_LBUTTONDOWN and len(lowers) == 0 :
      # Array
      # [
      #	[12 34 32],[231 33 42]
      # [22 33 44],[123 32 255]
      #]
      #
      pixel = img[y,x]

      # Divide to two subsections 
      upperThreshold = img[np.where(img > pixel)]
      lowerThreshold = img[np.where(img <= pixel)]
      
      # Calculate the mean from both
      lowerMean = np.average(lowerThreshold.flatten())
      upperMean = np.average(upperThreshold.flatten())
      lowerMean = lowerMean*COLOR_RANGE
      upperMean = upperMean*COLOR_RANGE
	
      print(lowerMean,upperMean)
      # Find the upper and lower
      upper =  np.array([int(pixel[0]+upperMean), int(pixel[1]+upperMean), int(pixel[2]+upperMean)])
      lower =  np.array([int(pixel[0]-lowerMean), int(pixel[1]-lowerMean), int(pixel[2]-lowerMean)])
      #lower =  np.array([int(pixel[0]-(finalMean)*pixel[0]), int(pixel[1]-(finalMean)*pixel[1]), int(pixel[2]-(finalMean)*pixel[2])])

      uppers.append(upper)
      lowers.append(lower)

def main():
    cam = cv2.VideoCapture("video.mp4")

    cv2.namedWindow(WINDOW_NAME)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        global img
        cv2.setMouseCallback(WINDOW_NAME, capture_click)

        oldX,oldY,oldW,oldH = -1,-1,-1,-1
        global lowers,uppers

        # Blurring
        blur = cv2.blur(frame,(1,1))
        blur0=cv2.medianBlur(blur,5)
        blur1= cv2.GaussianBlur(blur0,(1,1),0)
        blur2= cv2.bilateralFilter(blur1,9,200,200)

        # Sharping
        sharp=cv2.addWeighted(frame,3,blur2,-2,0)

        # Erosion
        kernel = np.ones((1,1),np.uint8)
        sharp = cv2.erode(sharp,kernel,iterations = 1)

        img = sharp
        if(len(lowers) > 0):
          curLow = lowers[0]
          curUpp = uppers[0]


          kernel = np.ones((1,1),np.uint8)
          sharp = cv2.erode(sharp,kernel,iterations = 1)
          cv2.imshow("er",sharp)

          # Create the mask
          mask = cv2.inRange(sharp,curLow,curUpp)

          cv2.imshow("mask",mask)
          # Find the contours
          contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

          if len(contours)>0:
            # Sort the contours
            cont_sort = sorted(contours, key=cv2.contourArea, reverse=True)
            area = max(cont_sort, key=cv2.contourArea)
            (xg,yg,wg,hg) = cv2.boundingRect(area)
            cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(69,69,255),2)
          cv2.imshow(WINDOW_NAME, frame)
        cv2.imshow(WINDOW_NAME, frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
