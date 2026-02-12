import os
import glob
import shutil

img_dir = r"D:\YOLO\yolo_all\dataset_org\images"
lab_dir = r"D:\YOLO\yolo_all\dataset_org\labels"

# 未匹配图片移动到这里
backup_dir = os.path.join(os.path.dirname(img_dir), "images_no_label_backup")
os.makedirs(backup_dir, exist_ok=True)

# 1) 收集所有 label 的基名（不含扩展名）
label_basenames = set()
for p in glob.glob(os.path.join(lab_dir, "*.txt")):
    base = os.path.splitext(os.path.basename(p))[0]
    label_basenames.add(base)

# 2) 遍历图片，把没有对应 label 的移走
img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
moved = 0
kept = 0
total = 0

for p in os.listdir(img_dir):
    if not p.lower().endswith(img_exts):
        continue
    total += 1
    base = os.path.splitext(p)[0]
    src = os.path.join(img_dir, p)

    if base not in label_basenames:
        dst = os.path.join(backup_dir, p)
        # 避免重名覆盖
        if os.path.exists(dst):
            name, ext = os.path.splitext(p)
            dst = os.path.join(backup_dir, f"{name}_dup{ext}")
        os.remove(src)
        # shutil.move(src, dst)
        moved += 1
    else:
        kept += 1

print(f"图片总数: {total}")
print(f"保留(有label): {kept}")
print(f"移走(无label): {moved}")
print("无label图片备份目录:", backup_dir)
print("清理完成！")