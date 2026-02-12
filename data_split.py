# 划分数据集，原始数据集放入dataset_org中，处理后的数据会生成在dataset中
import os
import shutil
import random

# 划分比例，分别是训练集、验证集合测试集，根据自己需要改一下
split_rate = [0.7, 0.2, 0.1]
split_names = ["train", "valid", "test"]

split_rate[1] = sum(split_rate[:2])

# 是否打乱
shuffle = True

def replace_expand_name(file_name, ex_name):
    return ".".join(file_name.split(".")[:-1] + [ex_name])

if os.path.exists("dataset"):
    shutil.rmtree("dataset")
os.makedirs("dataset")
for name in split_names:
    os.makedirs(os.path.join("dataset", name, "images"))
    os.makedirs(os.path.join("dataset", name, "labels"))

image_folder = r"dataset_org\images"
label_folder = r"dataset_org\labels"

image_files = os.listdir(image_folder)
if shuffle == True:
    random.shuffle(image_files)
label_files = [replace_expand_name(name, 'txt') for name in image_files]

def write_files(image_files, label_files, split):
    for image_file, label_file in zip(image_files, label_files):
        shutil.copy(os.path.join(image_folder, image_file), os.path.join("dataset", split, "images", image_file))
        shutil.copy(os.path.join(label_folder, label_file), os.path.join("dataset", split, "labels", label_file))

data_len = len(image_files)
write_files(image_files[:int(split_rate[0]*data_len)], label_files[:int(split_rate[0]*data_len)], split_names[0])
write_files(image_files[int(split_rate[0]*data_len):int(split_rate[1]*data_len)], label_files[int(split_rate[0]*data_len):int(split_rate[1]*data_len)], split_names[1])
write_files(image_files[int(split_rate[1]*data_len):], label_files[int(split_rate[1]*data_len):], split_names[2])

print("数据集划分完成！")


