import os

folder = r"D:\project\Yolov5\red_and_blue\val"   # 修改为你的图片目录

files = sorted(os.listdir(folder))

for i, file in enumerate(files, start=1):
    ext = os.path.splitext(file)[1]
    old_path = os.path.join(folder, file)
    new_path = os.path.join(folder, f"{i:04d}{ext}")

    os.rename(old_path, new_path)

print("重命名完成")