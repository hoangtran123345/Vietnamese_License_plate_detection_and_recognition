import cv2
import os

# Define constants
TARGET_WIDTH = 32
TARGET_HEIGHT = 32

FNT_FOLDER = "./English/English/Fnt/"
OUTPUT_ROOT_FOLDER = "./English_resize_new/"

# Create output root folder if it doesn't exist
os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)

# Process each subfolder from A to Z
for subfolder in range(ord('A'), ord('Z')+1):
    char_folder = os.path.join(FNT_FOLDER, chr(subfolder))

    # Skip non-existent folders
    if not os.path.exists(char_folder):
        continue

    output_folder = os.path.join(OUTPUT_ROOT_FOLDER, chr(subfolder))

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    i = 0
    # Process each image in the current folder
    for char_image in os.listdir(char_folder):
        char_path = os.path.join(char_folder, char_image)

        char_split = char_image

        # Read char image
        char = cv2.imread(char_path)

        # # Get row and column
        # rows, columns, _ = char.shape

        # # Calculate padding
        # paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        # paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # # Apply padding to make the image fit for the neural network model
        # char = cv2.copyMakeBorder(char, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])

        # Convert and resize image
        char = cv2.cvtColor(char, cv2.COLOR_BGR2RGB)
        char = cv2.resize(char, (TARGET_WIDTH, TARGET_HEIGHT))

        # Save the resized image
        OUTPUT_PATH = os.path.join(output_folder, char_split)
        cv2.imwrite(OUTPUT_PATH, char)
        i += 1

    print(f"Processed folder {chr(subfolder)} - Number of resized images: {i}")
