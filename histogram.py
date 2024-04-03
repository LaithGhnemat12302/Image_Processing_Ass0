import numpy as np
import cv2  # OpenCV Library
import matplotlib.pyplot as plt
# ________________________________________________________________________________________________________________
def histogram_equalization(original):
    # Convert the color image to grayScale
    if len(original.shape) == 3:   # If the image is colorful convert it to grayScale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Calculate histogram of the image
    hist, bins = np.histogram(original.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize CDF to scale it between 0 and 255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Interpolate the normalized CDF to get the new pixel values
    equalized_image = np.interp(original.flatten(), bins[:-1], cdf_normalized).reshape(original.shape)

    # Plot the histogram
    plt.hist(original.flatten(), 256, [0, 256], color='r')     # Original image
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='b')   # Equalized image
    plt.legend(['Original Histogram', 'Equalized Histogram'])
    plt.show()

    return equalized_image.astype(np.uint8)
# ________________________________________________________________________________________________________________
# Load an image
image = cv2.imread('lena.jpg')

# Apply histogram equalization
equalized_image = histogram_equalization(image)

# Display the original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()