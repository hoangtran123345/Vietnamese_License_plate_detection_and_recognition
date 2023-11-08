# Import the necessary packages
import argparse
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import functools
from skimage.filters import threshold_local
from crop_rotate import CropImageFromOriginal,rotate_and_crop


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--model", required=True, help="Path to the pre-trained model")
args = vars(ap.parse_args())

###############################################
# This takes two stages
# The first stage is to segment characters
# The second stage is to recognise characters
###############################################

###############################################
# The first stage
###############################################

# Read the image
image_root = (args["image"])
img = cv2.imread(image_root)
cv2.imshow("ANH GOC", img)

crop = CropImageFromOriginal(image_root)
image = rotate_and_crop(crop)
# Tach cac kenh mau trong anh HSV, chon kenh mau V(Value) do sang
if len(image.shape)==2:
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

# Tao 1 anh nguong cuc bo len kenh mau V
# 35: Block size kich thuoc cua so cuc bo, cang lon thi nguong cuc bo se duoc tinh toan tren mot khu vuc lon hon cua anh
# offset: do nhay cua nguong
# method: tao nguong su dung phuong phap gaussian 
T = threshold_local(V, 35, offset=5, method="gaussian")

# Ap dung nguong de tao anh nhi phan 
# Anh la 1 ma tran chi co 2 gia tri 0 va 255
thresh = (V > T).astype("uint8") * 255

# Dao nguoc anh, den thanh trang va nguoc lai
thresh = cv2.bitwise_not(thresh)
# cv2.imshow("thresh",thresh)


# Tim va danh dau cac thanh phan ket noi trong anh nhi phan
# Ham tra ve so thanh phan ket noi(KO QUAN TAM) va 1 ma tran labels co kich thuoc
# giong anh goc, moi pixel thuoc 1 thanh phan ket noi nhat dinh, thanh phan gia tri 0 la background
_, labels = cv2.connectedComponents(thresh)

# Tao ma tran kich thuoc giong anh thresh, toan anh den 
# mask dung de luu tru ket qua phan doan anh
mask = np.zeros(thresh.shape, dtype="uint8")
total_pixels = thresh.shape[0] * thresh.shape[1]    # Tinh tong pixels
lower = total_pixels // 90   # CU 120   # Nguong duoi
upper = total_pixels // 20              # Nguong tren

# Duyet qua cac gia tri duy nhat trong ma tran labels
for label in np.unique(labels):
    # Bo qua label background
    if label == 0:
        continue
    # Tao mask co cung kich thuoc voi anh nhi phan thresh
    labelMask = np.zeros(thresh.shape, dtype="uint8")  # anh den
    # Gan gia tri 255(mau trang) cho cac pixel thuoc ve thanh phan ket noi co 
    # gia tri la label
    labelMask[labels == label] = 255
    # Dem so pixel co gia tri khac 0 trong mask
    numPixels = cv2.countNonZero(labelMask)
    # Kiem tra so luong pixel co nam trong nguong xac dinh hay khong
    # Neu co thi add label vao mask
    if numPixels > lower and numPixels < upper:
        mask = cv2.add(mask, labelMask)


# Find contours and get bounding box for each contour
# Tìm các đường viền trong ảnh mask
# cv2.RETR_EXTERNAL chỉ trả về các đường viền ngoại vi, không chứa các đường viền con bên trong
# cv2.CHAIN_APPROX_SIMPLE là một phương pháp giảm độ phức tạp của đường viền để tiết kiệm không gian bộ nhớ
cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Duyet qua cac contours va tinh toan bb cho moi duong vien
# Ket qua la danh sach cac bb, moi bb duoc bieu dien duoi dang (x,y,w,h)
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
# print("Bounding boxes: ", boundingBoxes)
print("So bb cu la: ",len(boundingBoxes))
# Chuyen tu list sang mang numpy
boundingBoxes = np.array(boundingBoxes)
# Tim mean(Gia tri trung binh cua w,h,y)
mean_w = np.mean(boundingBoxes[:, 2])
mean_h = np.mean(boundingBoxes[:, 3])
mean_y = np.mean(boundingBoxes[:,1])
# Tim so luong boundingBoxes moi dua tren nguong trung binh chieu cao va chieu rong
threshold_w = mean_w * 1.5
threshold_h = mean_h * 1.5
new_boundingBoxes = boundingBoxes[(boundingBoxes[:, 2] < threshold_w) & (boundingBoxes[:, 3] < threshold_h)]
print("So bb moi la: ",len(new_boundingBoxes))


# Sort the bounding boxes from left to right, top to bottom
# sort by Y first, and then sort by X if Ys are similar
def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]
boundingBoxes = sorted(new_boundingBoxes, key=functools.cmp_to_key(compare) )


###############################################
# The second stage
###############################################

# Define constants
TARGET_WIDTH = 32
TARGET_HEIGHT = 32

# chars = [
#     '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
#     'H','K','L','M','N','P','R','S','T','U','V','X','Y','Z', 'ZNOISE'
#     ]

chars = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
      "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z", "ZNOISE"]

# Load the pre-trained convolutional neural network
model = load_model(args["model"], compile=False)


vehicle_plate = ""
# Loop over the bounding boxes
i = 0

# Resize image before show
NEW_WIDTH = 430
NEW_HEIGHT = 250
OLD_HEIGHT = image.shape[0]
OLD_WIDTH = image.shape[1]
image = cv2.resize(image,(NEW_WIDTH,NEW_HEIGHT))

# Ty le thay doi
scale_x = NEW_WIDTH / OLD_WIDTH
scale_y = NEW_HEIGHT / OLD_HEIGHT

for rect in boundingBoxes:

    # Get the coordinates from the bounding box
    x,y,w,h = rect

    # Crop the character from the mask
    # and apply bitwise_not because in our training data for pre-trained model
    # the characters are black on a white background
    crop = mask[y:y+h, x:x+w]
    crop = cv2.bitwise_not(crop)

    # Get the number of rows and columns for each cropped image
    # and calculate the padding to match the image input of pre-trained model
    rows = crop.shape[0]
    columns = crop.shape[1]
    paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
    paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)
    
    # Apply padding to make the image fit for neural network model
    crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])

    # Convert and resize image
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)     
    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
    # cv2.imshow(f"mask_{i}",crop)
    i=i+1


    # Prepare data for prediction
    crop = crop.astype("float") / 255.0
    crop = img_to_array(crop)
    crop = np.expand_dims(crop, axis=0)

    # Make prediction
    prob = model.predict(crop)[0]
    idx = np.argsort(prob)[-1]
    vehicle_plate += chars[idx]


    # New character bounding box
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)


    # Show bounding box and prediction on image
    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
    cv2.putText(image, chars[idx], (x-7,y+18), 0, 1.1, (0, 0, 255), 2)


# Show final image
cv2.imshow('Final', image)
print("Vehicle plate: " + vehicle_plate)
cv2.waitKey(0)