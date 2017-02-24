#**Finding Lane Lines on the Road** 

##Writeup 
Jabeer Ahmed

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

> **Canny Edge Detection**

The first step of the pipeline is to perform edge detection using Canny's method. This was already in the project template. I didn't change too many of the parameters to get it to my desired output state. 
The final parameters were:
	* `ED_kernel_size    = 5` 
	* `ED_low_threshold  = 50 `
	* `ED_high_threshold = 150`
 
 A sample output: 
 ![enter image description here](https://lh3.googleusercontent.com/-PttWf6GvylU/WK6b0S32kiI/AAAAAAAAyik/bAZwDpxjz_Q9YJeirjUoDfEjzLCRlnlGQCLcB/s0/raw_canny.png "Raw Canny edge points")
 

>  **Region Selection**

 Canny detector finds edges in high contrast areas. Therefore, the edge points shown above does not only limit to the lane line. In order to remove edges generator from the surrounding scenery, a quadrilateral region was selected as the mask as shown below. 

![enter image description here](https://lh3.googleusercontent.com/-DF2TmbGtedA/WK6c2fcG2AI/AAAAAAAAyi0/EMdDcs96RgsAxYErwa8Wyl4GaDBhulPDACLcB/s0/masked_canny.png "Region selected version of the Canny image")

The quad dimensions were :
	* `REG_top_width  = 0.1`  
	* `REG_btm_width  = 0.9`
	* `REG_top_height = 0.6` 

`REG_top_width, REG_btm_width` are the width of the top and bottom lines of the quad, respectively, expressed as a percentage of image width, centered around the image center. Similarly, `REG_top_height` is the height of the quad, expressed as a percentage of the image height and measured from the top of the image. 

>**Hough Lines**

Next, the hough line detection method was used to extract lines from cluster of points. Just like the previous two sets this was also provided along with the starter kit. Only some parameter tuning was required to get the desired result (below). The final parameters were: 	
    * `rho = 2` 	
    * `theta = np.pi/180 * 1` 
    * `threshold= 60` 
    * `min_line_length = 75`
    * `max_line_gap = 50`

![enter image description here](https://lh3.googleusercontent.com/-VJaYgufiYdI/WK6i9fyWdKI/AAAAAAAAyjM/0St8bctIIwM8WlNBr2cR6Tb-ekBZMBGNgCLcB/s0/13res_img.png "Hough Lines")

> **Gradient based Hough Lines Filtering**

After converting points to potential lines, a gradient based filter mechanism was applied to eliminate the spurious points. The spurious lines are often gradients closer to the x-axis, therefore those lines can easily be filtered out with a band-pass gradient filter. In order to get the band of gradients that correspond to the lane lines gradient, the entire pipeline runs once through both the example videos and keeps a log of all the lines detected. From this data set, a histogram of line gradient is created. The histogram is shown below.

![enter image description here](https://lh3.googleusercontent.com/-HBtlMZ5uIVI/WK_hifb1ktI/AAAAAAAAyjw/sYSEcyVmlkwAoUor0zdoamPFMi2xmSSoACLcB/s0/download.png "Gradient distribution")

The distribution shows clearly the 2 peaks, which correspond to the 2 lane lines. Conveniently the peaks are on the either side of the `gradient = 0` line. This information is used to separate the lines into left and right groups. Also, the mean and standard deviation of the respective groups is used to identify to 'band-pass' of gradients.

This is similar to training the system. In the first pass (or the initialization), the pipeline only continues till hough lines generation stage. It collects the line gradient distribution data to calibration the gradient band-pass parameters and the system is ready to go. This can be avoid from the actually pipeline by hard coding the parameters. However, in my implementation I've kept it there.

> **Create single line**

The final step is to take the many line segments that pass through the gradient filter and obtain a single line equation for each side of the lane. This final line equation is obtained by performing RANSAC on the final set of line points to find the best fit line.

 
###2. Identify potential shortcomings with your current pipeline

In no order:

 1. The RANSAC best fit line approach will not perform well for curved roads. May have to use 2nd order polynomial fitting to make the solution general enough to be used with both straight and curved road segments.
 2. Darker or shadowy lane regions may to work well under canny edge detection method in grayscale. Alternatively looking at chroma-based color space (YUV, HSV, LAB etc) may make the solution lumination invariant. 

###3. Suggest possible improvements to your pipeline

1. A vertical edge detector make work better than Canny detector, which is an isotropic detector. A vertical edge detector will be better at picking out vertical edges than other ones. Because lane lines are mostly vertical, a vertical edge detector might work better than canny. 
2. There is not history or velocity baked into model, meaning if we fail to detect sufficient lines due to poor road conditions, we cannot make predictions on the most probable lane line location. With some knowledge of the immediate few frames, we maybe mitigate brief data dropouts and reduce detection jitteriness to get a smoother lane lines.


