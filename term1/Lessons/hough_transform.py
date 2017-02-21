import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

COLOR_SWATCH = [ 
	[255, 64, 0],  [255, 128, 0], [255, 191, 0], 
	[255, 255, 0], [191, 255, 0], [128, 255, 0], [64, 255, 0],
	[0, 255, 0],   [0, 255, 64],  [0, 255, 128], 
	[0, 255, 191], [0, 255, 255], [0, 191, 255], 
	[0, 128, 255], [0, 64, 255],  [0, 0, 255], 
	[64, 0, 255],  [128, 0, 255], [191, 0, 255], 
	[255, 0, 255], [255, 0, 191], [255, 0, 128], 
	[255, 0, 64],  [255, 0, 0] 
]

def getColor(counter=0):
    return COLOR_SWATCH[(counter*5) % len(COLOR_SWATCH)]

def image_size(img):
    return (img.shape[0], img.shape[1])

def edge_detect(gray, kernel_size=5, low_threshold=50, high_threshold=150):
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges, blur_gray

def create_region_mask(imshape, vertices, ignore_mask_color = 255):
    mask = np.zeros((imshape[0], imshape[1]), dtype=np.uint8)        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask

def create_quad_region_mask(imshape, top_width = 0.1, top_height = 0.55, ignore_mask_color = 255):
    
    # This time we are defining a four sided polygon to mask
    h = imshape[0]
    w = imshape[1]
    w_min = ((1.0 - top_width)/2)*w
    w_max = ((1.0 + top_width)/2)*w
    h_min = (h * top_height)
    vertices = np.array([[(0, h), (w_min, h_min), (w_max, h_min), (w,h)]], dtype=np.int32)
    return vertices, create_region_mask(imshape, vertices)

def apply_mask(img, mask):
    
    a = image_size(img)
    b = image_size(mask)
    
    assert(a == b), "image(" + str(a) + ") and mask(" + str(b) + ") size mismatch"
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        masked_image = np.zeros_like(img)
        
        for i in range(channel_count):
            masked_image[:,:,i] = cv2.bitwise_and(img[:, :,i], mask)
        return masked_image
    else:
        return cv2.bitwise_and(img, mask)

def hough_lines_P(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines, multi_color=False, thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    i = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(     img, (x1, y1), (x2, y2), getColor(i), thickness)
            cv2.line(line_img, (x1, y1), (x2, y2), getColor(i), thickness)
        if (multi_color): i = i + 1
    return line_img

def process(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ## Detect Edges
    ED_kernel_size = 5; ED_low_threshold=50; ED_high_threshold=150
    [edges, blur_gray] = edge_detect(gray, ED_kernel_size, ED_low_threshold, ED_high_threshold)
    
    ## Create Region Mask
    REG_top_width = 0.1; REG_top_height = 0.60
    verts, mask = create_quad_region_mask(gray.shape, REG_top_width, REG_top_height)
    masked_edges = apply_mask(edges, mask)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 * 1 # angular resolution in radians of the Hough grid
    threshold = 60     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 75 #minimum number of pixels making up a line
    max_line_gap = 50    # maximum gap in pixels between connectable line segments
    
    gray_image = np.dstack((gray, gray, gray)) 
    lines = hough_lines_P(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)                                        
    line_image = draw_lines(gray_image, lines, thickness=3)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
    

#    org_plt = plt.figure(1).add_subplot(111)
#    org_plt.imshow(image)
#
#    roi_plt = plt.figure(2).add_subplot(111)
#    roi_plt.imshow(apply_mask(image, mask))    
#
#    lin_plt = plt.figure(3).add_subplot(111)
#    lin_plt.imshow(lines_edges)
#
#    gry_plt = plt.figure(4).add_subplot(111)
#    gry_plt.imshow(gray_image)
#    
#    edg_plt = plt.figure(5).add_subplot(111)
#    edg_plt.imshow(edges, cmap='gray')
#
#    plt.show()
    
    print("NumLines = " + str(len(lines)))
    
    return gray_image
    


#%% Run Test Images

test_dir = "../project1/test_images/"
test_files = [os.path.abspath(os.path.join(test_dir, f)) for f in os.listdir(test_dir)]                       

all_img = range(len(test_files))
sub_rng = [0, 1, 2, 3, 4, 5]
for image_index in all_img:
    image_name = test_files[image_index]
 
    ## Read in and grayscale the image
    image = mpimg.imread(image_name)
    
    process(image)
    
#%% Run White mp4
from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_dir = "../project1/"

white_output = os.path.join(test_dir, 'white.mp4')
clip1 = VideoFileClip( os.path.join(test_dir, "solidWhiteRight.mp4"))
white_clip = clip1.fl_image(process) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
<source src="{0}">
</video>
""".format(white_output))


#%% Run Yellow mp4
from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_dir = "../project1/"

yellow_output = os.path.join(test_dir, 'yellow.mp4')
clip2 = VideoFileClip( os.path.join(test_dir, "solidYellowLeft.mp4"))
yellow_clip = clip2.fl_image(process)
yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

