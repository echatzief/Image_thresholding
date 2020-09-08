import cv2
import numpy as np

WINDOW_NAME = "Video"
img = None

COLOR_RANGE=0.25

lowers = []
uppers = []

def determine_thresholds(event,x,y,flags,param):
	global img,lowers,uppers
	pixel = img[y,x]
	if event == cv2.EVENT_LBUTTONDOWN and len(lowers) == 0:
		
		# Array
		# [
		#	[12 34 32],[231 33 42]
		# [22 33 44],[123 32 255]
		#]
		#

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

	while True:
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break
		global img
		img = frame
		#cv2.imshow(WINDOW_NAME, frame)
		cv2.setMouseCallback(WINDOW_NAME, determine_thresholds)

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
			cv2.imshow("mask",mask)
			frame = cv2.bitwise_and(frame,frame,mask=mask)
		cv2.imshow(WINDOW_NAME,frame)
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
