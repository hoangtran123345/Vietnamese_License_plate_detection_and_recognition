import cv2
import shutil
import os
import random

CHAR_ROOT_FOLDER = "./QUAN_TRONG/Character_955_resize_new/"
ADDITIONAL_FOLDERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z", "ZNOISE"]
OUTPUT_ROOT_FOLDER = "./QUAN_TRONG/DATA_RESNET_2"
OUTPUT_SPLIT = ["train", "val", "test"]

os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)

for subfolder in ADDITIONAL_FOLDERS:
    char_folder = os.path.join(CHAR_ROOT_FOLDER,subfolder)

    for split in OUTPUT_SPLIT:
        if split == "train":
            train_folder = os.path.join(OUTPUT_ROOT_FOLDER,split)
            os.makedirs(train_folder, exist_ok=True)
            train_output = os.path.join(train_folder, subfolder)
            os.makedirs(train_output,exist_ok=True)
        if split == "val":
            val_folder = os.path.join(OUTPUT_ROOT_FOLDER,split)
            os.makedirs(val_folder, exist_ok=True)
            val_output = os.path.join(val_folder, subfolder)
            os.makedirs(val_output,exist_ok=True)
        if split == "test":
            test_folder = os.path.join(OUTPUT_ROOT_FOLDER,split)
            os.makedirs(test_folder, exist_ok=True)
            test_output = os.path.join(test_folder, subfolder)
            os.makedirs(test_output,exist_ok=True)

    all_images = os.listdir(char_folder)

    # B4: So luong anh trong moi phan
    total_size = len(all_images)
    train_size = int(0.8*total_size)
    val_size = int(0.1*total_size)

    # B5: Chia ngau nhien thanh cac tap train, val, test
    train_images = random.sample(all_images,train_size)
    remaining_images = [img for img in all_images if img not in train_images]
    val_images = random.sample(remaining_images,val_size)
    test_images = [img for img in all_images if img not in train_images and img not in val_images]

    # B5: Di chuyen anh toi cac thu muc tuong ung
    for img in train_images:
        shutil.move(os.path.join(char_folder,img), os.path.join(train_output,img))
    for img in val_images:
        shutil.move(os.path.join(char_folder,img), os.path.join(val_output,img))
    for img in test_images:
        shutil.move(os.path.join(char_folder,img), os.path.join(test_output,img))
