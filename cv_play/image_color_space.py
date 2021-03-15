import cv2
import numpy as np
from matplotlib import pyplot as plt

img_real = cv2.imread('realNahid.png')

print(type(img_real))

img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2LUV)
img_real = cv2.resize(img_real, (224, 224))

cv2.imshow("Real", img_real)

img_fake = cv2.imread('fakeNahid.png')

img_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2LUV)
img_fake = cv2.resize(img_fake, (224, 224))


cv2.imshow("Fake", img_fake)

# plt.hist(img_real.ravel(), 256, [0, 256])
# plt.show()

# plt.hist(img_fake.ravel(), 256, [0, 256])
# plt.show()

plt.hist(img_real.ravel(), 256, [0, 256], alpha=0.7, label='real')
plt.hist(img_fake.ravel(), 256, [0, 256], alpha=0.7, label='fake')
plt.legend(loc='upper right')
plt.show()

ret = cv2.imwrite("Gen.png", img_fake)
print(ret)

key = cv2.waitKey(0)
# if key & 0xFF == 27:
#     break

# # live stream
# cap = cv2.VideoCapture(0)

# while(cap.isOpened()):
#     try:
#         ret, frame = cap.read()
#         if ret:
#             print(ret)
#             # cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#             # cv2.resizeWindow('Image', 600, 500)
#         else:
#             print('Stream ended...')
#             # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             break
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     except Exception as ex:
#         print(ex)
