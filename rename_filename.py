""""
File ảnh gốc có 944 ảnh, đuôi .jpg, đặt tên dài, ko theo thứ tự
Đổi tên các file ảnh trên lần lượt theo thứ từ trên xuống dưới, với tên lần lượt từ car_0 -> car_943

Phương pháp 
1. Tìm đường dẫn đến folder chứa ảnh
2. Lấy ra tất cả các ảnh có đuôi .jpg
3. Sap xep danh sach theo thu tu tang dan
4. Duyệt qua từng tên file và doi tên

"""
import os
# Duong dan toi folder chua anh
folder_path = "./CarTGMT"

# Lấy danh sách tên tất cả các file trong thư mục kết thúc bằng ".jpg"
file_names = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

# Sắp xếp danh sách theo thứ tự tăng dần
file_names.sort()   # Phai co dong nay, van de lien quan den cach he thong sap xep ten file khi su dung os.listdir, 
                    # khi su dung .sort() dam bao rang danh sach duoc sap xep theo thu tu tang dan, giup cac ten file duoc doi theo 1 thu tu cu the, 
                    # con khi ko su dung thi thu tu cac tep tin trong danh sach co the phu thuoc vao cach he thong sap xep chung khi tra ve tu os.listdir, 
                    # do do viec doi ten co the xay ra theo thu tu ko dong deu va co the gay ra mat mat

# Duyet qua tung ten file va doi ten
for i, old_name in enumerate(file_names):
    new_name = f"car_{i}.jpg"
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed: {old_name} -> {new_name}")
    
print("Rename successful")
import torch.nn as nn

class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()

        # Định nghĩa các lớp convolutional và fully connected của mô hình VGG16
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)  # Đối với ImageNet, output là 1000 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
