import cv2
import numpy as np
img = None

COLOR_RANGE=18
lowers = []
uppers = []
pixels = []

# Retrieve the image pixel and define the lower and the upper
def capture_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img,lowers,uppers,pixels
        
        pixel = img[y,x]
        upper =  np.array([int(pixel[0]+(COLOR_RANGE/float(100))*pixel[0]), int(pixel[1]+(COLOR_RANGE/float(100))*pixel[1]), int(pixel[2]+(COLOR_RANGE/float(100))*pixel[2])])
        lower =  np.array([int(pixel[0]-(COLOR_RANGE/float(100))*pixel[0]), int(pixel[1]-(COLOR_RANGE/float(100))*pixel[1]), int(pixel[2]-(COLOR_RANGE/float(100))*pixel[2])])

        lowers.append(lower)
        uppers.append(upper)
        pixels.append(pixel)

def main():
    cam = cv2.VideoCapture("video.mp4")

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        global img
        img = frame
        cv2.setMouseCallback("test", capture_click)

        oldX,oldY,oldW,oldH = -1,-1,-1,-1
        global lowers,uppers
        for i in range(len(lowers)):
            curLow = lowers[i]
            curUpp = uppers[i]
            
            blur = cv2.blur(frame,(1,1))
            blur0=cv2.medianBlur(blur,5)
            blur1= cv2.GaussianBlur(blur0,(1,1),0)
            blur2= cv2.bilateralFilter(blur1,9,200,200)

            mask = cv2.inRange(blur2,curLow,curUpp)
            contours = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
            frame = cv2.bitwise_and(frame,frame,mask=mask)

        cv2.imshow("test", frame)

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
