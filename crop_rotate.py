import torch
import numpy as np
import cv2
from imutils import perspective
import numpy as np
from skimage.filters import threshold_local
import imutils
from keras.models import load_model
import os

best_path = "yolov5_license_plate_600.pt"
model_detect_frame = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=best_path, force_reload=True)

def rotate_and_crop(img):
    # Chuyen doi anh sang dang den trang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(f"gray image {file_name}.jpg",gray)

    # Nhi phan hoa anh su dung nguong OTSU: Tu xac dinh nghuong toi uu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    

    # Tim tat ca cac contours: duong vien cua anh da duoc nhi phan hoa
    # Mỗi first_contour là một mảng numpy với các kích thước là (n, 1, 2), 
    # trong đó n là số lượng điểm trên đường viền và (1, 2) là hình dạng của mỗi điểm (một hàng và hai cột cho tọa độ x và y).
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(thresh,contours,(255,0,0),1)
    
 
    # Tim ra max contours bang cach tinh dien tich coutours bao phu
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    # print("maxt_cnt",max_cnt)
    thresh_copy = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(thresh_copy,max_cnt,-1,(255,0,0),2)
    # cv2.imshow(f"thresh image {file_name}.jpg", thresh_copy)
    # x,y,w,h = cv2.boundingRect(max_cnt)
    
    # Tim hinh chu nhat co dien tich nho nhat bao quanh contours lon nhat
    rect = cv2.minAreaRect(max_cnt)
    # (cx,cy): toa do tam cua hinh chu nhat     (cw,ch): chieu dai chieu rong cua hinh chu nhat
    # angle duoc xac dinh bang truc chinh cua hcn(doan nam doc theo chieu dai) va truc ngang cua anh, theo chieu kim dong ho
    # nếu giá trị góc xoay là 30 độ, nghĩa là hình chữ nhật được quay theo chiều kim đồng hồ 30 độ so với trục ngang của ảnh
    ((cx,cy),(cw,ch),angle) = rect
    # print(f"Angle image {file_name}.jpg co gia tri: ", angle)
    if angle < 45:
        M = cv2.getRotationMatrix2D((cx,cy), angle, 1)
    else:
        M = cv2.getRotationMatrix2D((cx,cy), angle-90, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    # return rotated
    
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x,y,w,h = cv2.boundingRect(max_cnt)
    cropped = rotated[y:y+h, x:x+w]    # NEW
    # cv2.imshow(f"cropped and rotate image {file_name}.jpg",cropped)
    # output_path = f"./Data_Crop_Rotate/TEST_ANH_MANG_CROP_ROTATE/{file_name}.jpg"
    # cv2.imwrite(output_path,cropped)
    return cropped

def CropImageFromOriginal(imgPath):
    image = cv2.imread(imgPath)
    # cv2.imshow(f"imageOri {file_name}.jpg",image)
    results = model_detect_frame(image)
    df = results.pandas().xyxy[0]
    for obj in df.iloc:
        xmin = float(obj['xmin'])
        xmax = float(obj['xmax'])
        ymin = float(obj['ymin'])
        ymax = float(obj['ymax'])
    coord = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    crop = perspective.four_point_transform(image, coord)
    # cv2.imshow(f"Crop image {file_name}.jpg", crop)
    return crop



FOLDER_PATH = './Data_Crop_Rotate/TEST_ANH_MANG'
i=0
for name in os.listdir(FOLDER_PATH):
    img_path = os.path.join(FOLDER_PATH,name)
    file_name_jpg = img_path.split("/")[-1]
    file_name = file_name_jpg.split(".")[0]
# print("file_name: ", file_name)

    crop = CropImageFromOriginal(img_path)
    rotate_and_crop(crop)
    i += 1
# cv2.waitKey(0)
print("So luong anh da crop va rotate la: ",i)

