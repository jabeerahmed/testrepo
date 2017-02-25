#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import Helper

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import os
os.listdir("test_images/")


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def color_selection(image):
	# Define color selection criteria
	# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
	red_threshold = 200
	green_threshold = 200
	blue_threshold = 200
	rgb_threshold = [red_threshold, green_threshold, blue_threshold]

	color_select = np.copy(image)
	# Do a boolean or with the "|" character to identify
	# pixels below the thresholds
	thresholds =  (image[:,:,0] < rgb_threshold[0]) \
	            | (image[:,:,1] < rgb_threshold[1]) \
	            | (image[:,:,2] < rgb_threshold[2])
	color_select[thresholds] = [0,0,0]
	return color_select




def process_image(image):
	# NOTE: The output you return should be a color image (3 channel) for processing video below
	# TODO: put your pipeline here,
	# you should return the final output (image where lines are drawn on lanes)

	# Grab the x and y size and make a copy of the image
	ysize = image.shape[0]
	xsize = image.shape[1]
	color_select = np.copy(image)
	line_image = np.copy(image)

	rgb_threshold = color_selection(image)

	plt.imshow(rgb_threshold)
	plt.show()

	return image

process_image(image)


# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(white_output))


# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(yellow_output))