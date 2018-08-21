import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_path = input("Enter the image path\n")
slash_index = file_path.rfind('/')

if slash_index == -1:
    file_full_name = file_path
else:
    file_extra_path = file_path[:slash_index]
    file_full_name = file_path[slash_index+1:]

dot_index = file_full_name.rfind('.')
file_name = file_full_name[:dot_index]
file_extension = file_full_name[dot_index+1:]

input_img = mpimg.imread(file_path)

output_img = cv2.fastNlMeansDenoisingColored(input_img, None, 10, 10, 7, 21)

output_file_name = file_name + "_denoised"
output_file_full_name = output_file_name + '.' + file_extension
output_file_path = output_file_full_name if slash_index == -1 else file_extra_path + '/' + output_file_full_name

plt_file_name = file_name + "_plt"
plt_file_full_name = plt_file_name + '.' + file_extension
plt_file_path = plt_file_full_name if slash_index == -1 else file_extra_path + '/' + plt_file_full_name

mpimg.imsave(output_file_path, output_img)

plt.subplot(121), plt.imshow(input_img)
plt.subplot(122), plt.imshow(output_img)
plt.savefig(plt_file_path)
plt.show()