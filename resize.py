import cv2
import os

# Define constants
TARGET_WIDTH = 32
TARGET_HEIGHT = 32

CHAR_FOLDER = "./Character/ZNOISE"
for char_image in os.listdir(CHAR_FOLDER):
    char_path = os.path.join(CHAR_FOLDER,char_image)

    char_split = char_path.split("/")[-1]
    # char_file_name = char_split.split(".")[0]

    # Read char image
    char = cv2.imread(char_path)
    print(f"Shape {char_split} before", char.shape)
    # cv2.imshow(f"{char_split} before", char)
    char = cv2.bitwise_not(char)
    # Get row and columb
    rows = char.shape[0]
    columns = char.shape[1]

    paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
    paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

    # Apply padding to make the image fit for neural network model
    char = cv2.copyMakeBorder(char, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])

    # Convert and resize image
    char = cv2.cvtColor(char, cv2.COLOR_BGR2RGB)     
    char = cv2.resize(char, (TARGET_WIDTH, TARGET_HEIGHT))
    print(f"Shape {char_split} after", char.shape)
    # cv2.imshow("Char after", char)

    OUTPUT_PATH = f"./DATA_CNN_128/ZNOISE/{char_split}"
    cv2.imwrite(OUTPUT_PATH,char)
    # cv2.waitKey(0)