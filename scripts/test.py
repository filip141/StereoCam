import cv2
import sys

cap2 = cv2.VideoCapture(1)
cap2.set(3, 320)
cap2.set(4, 240)

while (1):

    _, frame_r = cap2.read()

    print sys.stdout.write(frame_r.tostring())

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
