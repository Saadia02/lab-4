import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu
from scipy.stats import linregress
import glob
import math

time = 0
time_list = []
area_list = []

# Update the path and pattern based on your file structure
path = "D:\image processing\myenv\Lab-4\*.*"

# Print the files returned by glob.glob() for debugging
files = glob.glob(path)
print("Files found:", files)

num_images = len(files)

# Calculate the number of rows and columns for the subplot grid
num_cols = int(math.ceil(math.sqrt(num_images)))
num_rows = int(math.ceil(num_images / num_cols))

# Create a figure and a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

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

    # Calculate the subplot index
    row_idx = idx // num_cols
    col_idx = idx % num_cols

    # Display the original image in the left subplot
    axes[row_idx, col_idx].imshow(img, cmap='gray')
    axes[row_idx, col_idx].set_title(f'Time = {time} hr\nOriginal Image')

    # Display the thresholded image in the right subplot with a different color
    axes[row_idx, col_idx + 1].imshow(binary, cmap='coolwarm', interpolation='none')
    axes[row_idx, col_idx + 1].set_title('Thresholded Image')

    # Move to the next row if we've filled the current one
    if col_idx + 2 >= num_cols:
        row_idx += 1

# Remove any empty subplots
for idx in range(num_images, num_rows * num_cols):
     axes.flatten()[idx].remove()

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

plt.plot(time_list, area_list, 'bo')  # Print blue dots scatter plot

slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list)
print("y = ", slope, "x", " + ", intercept)
print("R² = ", r_value**2)