import cv2
import numpy as np

img=cv2.imread("F_2_Color_Color.png")

cv2.imshow("frame",np.array(img))
cv2.waitKey(0)


# cv2.destroyAllWindows()