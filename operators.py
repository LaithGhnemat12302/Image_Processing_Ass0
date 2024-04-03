import numpy as np
import cv2
# ________________________________________________________________________________________________________________
def histogram_equalization(original):
    # Convert the color image to grayScale
    if len(original.shape) == 3:  # If the image is colorful convert it to grayScale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Calculate histogram of the image
    hist, bins = np.histogram(original.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function (CDF) of the histogram
    cdf = hist.cumsum()

    # Normalize CDF to scale it between 0 and 255
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    # Interpolate the normalized CDF to get the new pixel values
    equalized_image = np.interp(original.flatten(), bins[:-1], cdf_normalized).reshape(original.shape)

    return equalized_image.astype(np.uint8)
# ________________________________________________________________________________________________________________
def contrast_stretching(original, r1, s1, r2, s2):
    # Convert the color image to grayScale
    if len(original.shape) == 3:  # If the image is colorful convert it to grayScale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply contrast stretching
    result_image = np.piecewise(original, [original < r1, (original >= r1) & (original <= r2), original > r2],
                                [lambda x: s1 / r1 * x, lambda x: ((s2 - s1) / (r2 - r1)) * (x - r1) + s1,
                                 lambda x: ((255 - s2) / (255 - r2)) * (x - r2) + s2])

    return result_image.astype(np.uint8)
# ________________________________________________________________________________________________________________
def log_operator(original, c):
    # Convert the color image to grayScale
    if len(original.shape) == 3:  # If the image is colorful convert it to grayScale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply log operator
    result_image = c * np.log1p(original)

    return result_image.astype(np.uint8)
# ________________________________________________________________________________________________________________
def point_processing(original, alpha, beta):
    # Convert the color image to grayScale
    if len(original.shape) == 3:  # If the image is colorful convert it to grayScale
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply point processing
    result_image = alpha * original + beta

    # Clip the pixel values to ensure they are within the valid range [0, 255]
    result_image = np.clip(result_image, 0, 255)

    return result_image.astype(np.uint8)
# ________________________________________________________________________________________________________________
# Load an image
original = cv2.imread('lena.jpg')

# Apply histogram equalization
equalized_image = histogram_equalization(original)

# Apply contrast stretching
stretched_image = contrast_stretching(original, 50, 0, 150, 255)

# Apply log operator
log_enhanced_image = log_operator(original, 10)  # brighter

# Apply point processing
point_processed_image = point_processing(original, 1.5, 30)

# Display the original, equalized, stretched, log-enhanced, and point-processed images
cv2.imshow('Original Image', original)
cv2.imshow('Equalized Image', equalized_image)
cv2.imshow('Stretched Image', stretched_image)
cv2.imshow('Log-Enhanced Image', log_enhanced_image)
cv2.imshow('Point-Processed Image', point_processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()