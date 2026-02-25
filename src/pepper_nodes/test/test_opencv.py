import numpy as np
import cv2

image= np.full((300, 300), 255, np.uint8)
cv2.imshow("Pepper Camera", image)
cv2.waitKey()