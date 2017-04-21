import numpy as np
import cv2
from matplotlib import pyplot as plt

file_name = 'noised/fest2'
file_extension = ".jpeg"
img = cv2.imread(file_name + file_extension)

dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

file_output = "denoised/" + file_name + "_denoised" + file_extension
cv2.imwrite(file_output, dst)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()