import cv2
import numpy as np
img = None

COLOR_RANGE=18
lowers = []
uppers = []
def capture_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global img,lowers,uppers
        pixel = img[y,x]

        print(x,y)
        upper =  np.array([int(pixel[0]+(COLOR_RANGE/float(100))*pixel[0]), int(pixel[1]+(COLOR_RANGE/float(100))*pixel[1]), int(pixel[2]+(COLOR_RANGE/float(100))*pixel[2])])
        lower =  np.array([int(pixel[0]-(COLOR_RANGE/float(100))*pixel[0]), int(pixel[1]-(COLOR_RANGE/float(100))*pixel[1]), int(pixel[2]-(COLOR_RANGE/float(100))*pixel[2])])

        print(upper,lower)
        lowers.append(lower)
        uppers.append(upper)

def main():
    cam = cv2.VideoCapture(0)

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


        global lowers,uppers
        for i in range(len(lowers)):
            curLow = lowers[i]
            curUpp = uppers[i]
            
            print(curLow,curUpp)
            mask = cv2.inRange(frame,curLow,curUpp)
            im2, contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                c = max(contours, key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(69,69,255),2)
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
