pkg load image
im = imread("test_images/solidWhiteCurve.jpg");
imGray = rgb2gray(im);

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
