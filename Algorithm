from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
from skimage import io
import cv2
from google.colab.patches import cv2_imshow  # Import the Colab patch for cv2.imshow
import matplotlib.pyplot as plt
from PIL import Image
# Define the window_and_level function
def window_and_level(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = np.clip(np.array(image), img_min, img_max)
    windowed_image = (windowed_image - img_min) / (img_max - img_min) * 255
    return Image.fromarray(windowed_image.astype(np.uint8))
# Define a function to process images from a folder
def process_images_from_folder1(clean_input_folder1):
    # Get a list of image files in the folder
    clean_image_files1 = [f for f in os.listdir(clean_input_folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # Process each image
    for clean_image_file1 in clean_image_files1:
        clean_image_path1 = os.path.join(clean_input_folder1, clean_image_file1)
        clean_image1 = Image.open(clean_image_path1)
        # Convert the image to grayscale
        image_gray1 = clean_image1.convert('L')
        # Window and level parameters
        window_center = 100
        window_width = 120
        # Apply window and level adjustments
        image_windowed1 = window_and_level(image_gray1, window_center, window_width)
        # Save the processed image
        # Creating the output folder if it doesn't exist
        clean_output_folder1 = "/content/drive/MyDrive/clean_crystal_images" # Moving the output folder variable inside the function
        os.makedirs(clean_output_folder1, exist_ok=True) # Creating the directory if it doesn't exist
        clean_output_path1 = os.path.join(clean_output_folder1, f"processed_{clean_image_file1}")
        image_windowed1.save(clean_output_path1)
        print(f"Processed image saved as {clean_output_path1}")
# calling the function by passing the path
clean_input_folder1 = "/content/drive/MyDrive/Crystal"
process_images_from_folder1(clean_input_folder1)
# Get a list of image files in the folder
test_image_files1 = [f for f in os.listdir("/content/drive/MyDrive/clean_crystal_images") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
# Process each image
for test_image_file1 in test_image_files1:
    test_image_path1 = os.path.join("/content/drive/MyDrive/clean_crystal_images", test_image_file1)
    test_image1 = Image.open(test_image_path1)
    #example_image = io.imread(example_image_path)
    example_image = io.imread(test_image_path1)
    if len(example_image.shape) == 2:
        example_image = cv2.cvtColor(example_image, cv2.COLOR_GRAY2RGB)
    # Preprocess the example image for better edge detection
    gray = cv2.cvtColor(example_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in x direction
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in y direction
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    # Apply binary thresholding for better edge detection
    _, binary_thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    # Morphological operations to remove small noises and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter and classify crystals
    bounding_box_image = example_image.copy()
    min_area = 5000  # Minimum area threshold for crystals
    max_area = 100000  # Maximum area threshold for crystals
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
            # Draw the bounding box
            crystal_size = area
            # Assign colors based on crystal size
            if crystal_size < 10000:
                color = (255, 0, 0)  # Blue for small crystals
            elif 10000 <= crystal_size < 20000:
                color = (0, 255, 0)  # Green for medium crystals
            else:
                color = (0, 0, 255)  # Red for large crystals
            # Draw the rectangle
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), color, 2)
            # Put the text indicating the width and height within the bounding box
            text = f"{w}x{h}"
            font_scale = 0.7  # Increase the font scale for larger text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(bounding_box_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    # Show the modified image with bounding boxes
    # cv2_imshow(bounding_box_image)
    # Save the modified image with bounding boxes
    bounded_output_folder = '/content/drive/MyDrive/BoundedImagesArea'
    os.makedirs(bounded_output_folder, exist_ok=True)
    bounded_output_path = os.path.join(bounded_output_folder, test_image_file1)
    # bounding_box_image.save(bounded_output_path)
    cv2.imwrite(bounded_output_path, bounding_box_image)
    # Print a success message
    print(f"Bounded image saved as {bounded_output_path}")
cv2.waitKey(0)
cv2.destroyAllWindows()
