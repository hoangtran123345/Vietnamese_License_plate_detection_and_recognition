import functools
import cv2
from skimage.filters import threshold_local
import numpy as np
import imutils
import os

FOLDER_PATH = './Data_Crop_Rotate/GreenParking_Crop_Rotate'
for name in os.listdir(FOLDER_PATH):
    image_path = os.path.join(FOLDER_PATH,name)
    # image_path = "./crop_rotate_1/car_93_crop_rotate.jpg"
    file_name_jpg = image_path.split("/")[-1]
    file_name = file_name_jpg.split(".")[0]
    image = cv2.imread(image_path)

    # Tach cac kenh mau trong anh HSV, chon kenh mau V(Value) do sang
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
    # # Xac dinh bounding box thuoc line nao, su dung y va mean_y
    # line1 = []
    # line2 = []
    # for box in new_boundingBoxes:
    #     x,y,w,h  = box
    #     if y > mean_y * 1.2:
    #         line2.append(box)
    #     else:
    #         line1.append(box)

    # # Sap xep cac boundingBoxes line1 va line2 dua tren toa do x cua goc trai tren cung box[0]
    # line1 = sorted(line1, key=lambda box: box[0])
    # line2 = sorted(line2, key=lambda box: box[0])
    # boundingBoxes = line1 + line2


    # Vẽ bounding boxes lên ảnh 
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # img_with_boxes = imutils.resize(image.copy(), width=600)
    # image = imutils.resize(image.copy(), width=600)

    # Đường dẫn thư mục chính
    OUTPUT = "./Data_Character"
    sub_folder = "GreenParking_Crop_Rotate"
    # Tạo đường dẫn đầy đủ đến thư mục con 2_1, 2_2, 2_3
    subfolder_path = os.path.join(OUTPUT, sub_folder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
    
    img_folder = file_name
    img_folder_path = os.path.join(subfolder_path,img_folder)
    if not os.path.exists(img_folder_path):
        os.mkdir(img_folder_path)


    i=0
    for bbox in boundingBoxes:
        x, y, w, h = bbox
        mask_n = mask[y:y+h, x:x+w]
        # cv2.imshow(f"box_{i}",mask_n)
        output_path = f"{img_folder_path}/{file_name}_{i}.jpg"
        cv2.imwrite(output_path,mask_n)
        i += 1
        # cv2.rectangle(mask_rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # cv2.imshow("image", image)
    # cv2.imshow("thresh",thresh)
    # cv2.imshow("mask",mask)
    # cv2.imshow("img_with_boxes",mask_rgb)
    # cv2.waitKey(0)
