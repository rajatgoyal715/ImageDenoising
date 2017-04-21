import numpy as np
import cv2
from matplotlib import pyplot as plt

file_name = 'fest'
file_extension = ".jpeg"
file_input = file_name + file_extension
img = cv2.imread(file_input)

dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

cv2.imwrite(file_name + "_denoised" + file_extension, dst)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()