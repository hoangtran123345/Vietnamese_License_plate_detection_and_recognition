"""
Muc tieu: + Data foder CarTGMT gom 944 anh duoi .jpg chia thanh 3 folder train,val,test
          + Ty le 8:1:1
          + Chia random, ko co thu tu

"""
# Cac buoc thuc hien:

import os
import random
import shutil

# B1: Duong dan toi thu muc chua anh
source_folder = "./CarTGMT"

# B2: Duong dan toi cac thu muc train, val, test
train_folder = "./dataset_car/train"
val_folder = "./dataset_car/val"
test_folder = "./dataset_car/test"

# B3: Danh sach tat ca cac tep tin anh
all_images = os.listdir(source_folder)

# B4: So luong anh trong moi phan
total_size = len(all_images)
train_size = int(0.8*total_size)
val_size = int(0.1*total_size)
print("TOTAL SIZE:", total_size)
# print("TRAIN SIZE:", train_size)
# print("VAL SIZE:", val_size)

# B5: Chia ngau nhien thanh cac tap train, val, test
train_images = random.sample(all_images,train_size)
remaining_images = [img for img in all_images if img not in train_images]
val_images = random.sample(remaining_images,val_size)
test_images = [img for img in all_images if img not in train_images and img not in val_images]
print("TRAIN SIZE:", len(train_images))
print("VAL SIZE:", len(val_images))
print("TEST SIZE:", len(test_images))

# B5: Di chuyen anh toi cac thu muc tuong ung
for img in train_images:
    shutil.move(os.path.join(source_folder,img), os.path.join(train_folder,img))
for img in val_images:
    shutil.move(os.path.join(source_folder,img), os.path.join(val_folder,img))
for img in test_images:
    shutil.move(os.path.join(source_folder,img), os.path.join(test_folder,img))