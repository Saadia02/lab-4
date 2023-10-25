import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu
from scipy.stats import linregress
import glob

time = 0
time_list = []
area_list = []

# Update the path and pattern based on your file structure
path = "D:\image processing\myenv\Lab-4\*.*"

# Print the files returned by glob.glob() for debugging
files = glob.glob(path)
print("Files found:", files)

num_images = len(files)

# Create a figure and a grid of subplots
fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

for idx, file in enumerate(files):
    img = io.imread(file)
    entropy_img = entropy(img, disk(3))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    scratch_area = np.sum(binary == 1)
    print("time=", time, "hr  ", "Scratch area=", scratch_area, "pix²")
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1

    # Display the original image in the left column
    axes[idx, 0].imshow(img, cmap='gray')
    axes[idx, 0].set_title(f'Time = {time} hr\nOriginal Image')

    # Display the thresholded image in the right column with a different color
    axes[idx, 1].imshow(binary, cmap='coolwarm', interpolation='none')
    axes[idx, 1].set_title('Thresholded Image')

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

plt.plot(time_list, area_list, 'bo')  # Print blue dots scatter plot

slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list)
print("y = ", slope, "x", " + ", intercept)
print("R² = ", r_value**2)
plt.show()