import os
import glob
from PIL import Image

# 你的路径：labels + images
label_dir = r"D:\YOLO\yolo_all\dataset_org\labels"
image_dir = r"D:\YOLO\yolo_all\dataset_org\images"  # 改成你的图片目录

class_map = {
    "Car": 0,
    "Cyclist": 1,
    "DontCare": 2,
    "Pedestrian": 3,
    "bicycle": 4,
    "bicycle_rack": 5,
    "human_depiction": 6,
    "moped_scooter": 7,
    "motor": 8,
    "ride_other": 9,
    "ride_uncertain": 10,
    "rider": 11,
    "truck": 12,
    "vehicle_other": 13
}

# KITTI/VOD bbox字段位置（0-based）：left top right bottom
IDX_L, IDX_T, IDX_R, IDX_B = 4, 5, 6, 7

img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image(base_name: str) -> str | None:
    for ext in img_exts:
        p = os.path.join(image_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None

total_files = 0
total_boxes = 0
skipped_dontcare = 0
skipped_unknown = 0
skipped_badline = 0
skipped_noimage = 0
skipped_invalid_box = 0

for file in glob.glob(os.path.join(label_dir, "*.txt")):
    total_files += 1
    base = os.path.splitext(os.path.basename(file))[0]
    img_path = find_image(base)

    if img_path is None:
        skipped_noimage += 1
        # 没图就不做归一化转换，跳过该txt（你也可以选择清空它）
        continue

    W, H = Image.open(img_path).size

    new_lines = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls_name = parts[0]

            # 跳过 DontCare
            if cls_name == "DontCare":
                skipped_dontcare += 1
                continue

            if cls_name not in class_map:
                skipped_unknown += 1
                continue

            # 必须至少有到bbox的8列
            if len(parts) <= IDX_B:
                skipped_badline += 1
                continue

            try:
                cls_id = class_map[cls_name]
                left = float(parts[IDX_L])
                top = float(parts[IDX_T])
                right = float(parts[IDX_R])
                bottom = float(parts[IDX_B])
            except:
                skipped_badline += 1
                continue

            # 纠正顺序（避免 left>right / top>bottom）
            x1, x2 = (left, right) if left <= right else (right, left)
            y1, y2 = (top, bottom) if top <= bottom else (bottom, top)

            # 裁剪到图像范围
            x1 = max(0.0, min(x1, W))
            x2 = max(0.0, min(x2, W))
            y1 = max(0.0, min(y1, H))
            y2 = max(0.0, min(y2, H))

            bw = x2 - x1
            bh = y2 - y1
            if bw < 2 or bh < 2:
                skipped_invalid_box += 1
                continue

            xc = (x1 + x2) / 2.0 / W
            yc = (y1 + y2) / 2.0 / H
            bw = bw / W
            bh = bh / H

            # 合法性检查
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                skipped_invalid_box += 1
                continue

            new_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            total_boxes += 1

    # 覆盖写回：只保留 YOLO 5列
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

print("完成：VoD/KITTI -> YOLOv8 labels（5列）")
print("txt文件数:", total_files)
print("输出框数:", total_boxes)
print("跳过 DontCare 行:", skipped_dontcare)
print("跳过未知类:", skipped_unknown)
print("坏行/解析失败:", skipped_badline)
print("没有对应图片的txt:", skipped_noimage)
print("无效框(过小/越界等):", skipped_invalid_box)
