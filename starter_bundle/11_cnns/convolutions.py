from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def rescale_intensity_personal(image: np.ndarray):
    """Limits image intensities to the range 0 to 255.
    If the value lies outside the range then it is set
    to the range value that is closest i.e. 0 or 255"""

    img_min = np.min(image)
    img_range = np.max(image) - img_min
    image = (image - img_min) / img_range
    return image * 255


def convolve(image, K):
    # get spacial dims of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for output img and pad boarders so
    # spatial size (i.e. width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # loop over input img, sliding the kernel across the img
    # from left to right and top to bottom
    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            # extract ROI (Region Of Interest) of img by extracting *center*
            # region of current coord dims
            roi = image[y-pad: y+pad+1, x-pad: x+pad+1]

            # perform convolution (or cross-correlation)
            k = (roi * K).sum()

            # store convolved value
            output[y-pad, x-pad] = k

    # rescale the output image to be in the range [0, 255]
    # noinspection PyTypeChecker
    output = rescale_intensity(output, in_range=(0, 255))
    # noinspection PyTypeChecker
    output = (output * 255).astype(np.uint8)

    return output


# commandline args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# construct avg blurring kernels used to smooth and image
small_blur = np.ones((7, 7), dtype="float") * (1.0 / (7*7))
large_blur = np.ones((21, 21), dtype="float") * (1.0 / (21*21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")

# construct Laplacian kernel used to detect edge-like regions of img
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
), dtype="int")

# construct Sobel x-axis kernel to detect x-axis edge-like regions
sobel_x = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
), dtype="int")

# construct Sobel x-axis kernel to detect y-axis edge-like regions
sobel_y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
), dtype="int")

# construct an emboss kernel
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
), dtype="int")

kernel_bank = (
    ("small_blur", small_blur),
    ("large_blur", large_blur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobel_x),
    ("sobel_y", sobel_y),
    ("emboss", emboss)
)

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for (kernel_name, Kernel) in kernel_bank:
    print(f"[INFO] applying {kernel_name} kernel")
    convolve_output = convolve(gray, Kernel)
    opencv_output = cv2.filter2D(gray, -1, Kernel)

    cv2.imshow("Original", gray)
    cv2.imshow(f"{kernel_name} - convolve", convolve_output)
    cv2.imshow(f"{kernel_name} - opencv", opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
