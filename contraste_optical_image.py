import polsar_basics as pb
import polsar_plot as pplot
#
import matplotlib.pyplot as plt
import numpy as np
import cv2
#
img_path = "./Data/Img10.jpg"
image = cv2.imread(img_path)
print(image)
print(type(image))
#img = img_path
plt.imshow(image)
plt.show()
#image_cont = pplot.image_contrast_brightness_optical(image)
contrast = 2.3
brightness = 0
image_cont = cv2.convertScaleAbs(image, alpha= contrast, beta=brightness)
plt.imshow(image_cont)
plt.show()
#
image_name = "img_10_contrast"
pplot.show_image_pauli_to_file_set_unless_img(image_cont,image_name)
