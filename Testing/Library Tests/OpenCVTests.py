"""
Created on Wed Jun 29 22:05:08 2022
@author: Lance
"""
import cv2
import matplotlib.pyplot                         as plt

light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)



nemo = cv2.imread("clownfish.jpg")


plt.imshow(nemo)
plt.show()


nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

plt.imshow(nemo)
plt.show()

hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)

result = cv2.bitwise_and(nemo, nemo, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()