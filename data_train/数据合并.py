import os
import csv
import shutil
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 记录当前工作目录，并切换到项目根目录，以便 config.py 能正确加载其资源文件
original_cwd = os.getcwd()
os.chdir(project_root)

# 将项目根目录添加到 sys.path 以便导入 config
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import MONSTER_COUNT, FIELD_FEATURE_COUNT
finally:
    # 切换回原始工作目录
    os.chdir(original_cwd)

def get_expected_header():
    """根据 start_auto_fetch 的逻辑生成预期表头"""
    if FIELD_FEATURE_COUNT > 0:
        header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}LF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}RF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += ["Result", "ImgPath"]
    else:
        header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
        header += ["Result", "ImgPath"]
    return header

def read_csv_data(filepath):
    """读取CSV文件，自动检测编码并返回表头和数据行"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, [], encoding
                data = list(reader)
                return header, data, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法以支持的编码读取文件 {filepath}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_images_dir = os.path.join(base_dir, 'images')
    target_csv_path = os.path.join(base_dir, 'arknights.csv')

    os.makedirs(target_images_dir, exist_ok=True)

    expected_header = get_expected_header()
    merged_data = []

    # 扫描子目录
    scan_dirs = [base_dir]
    tmp_dir = os.path.join(base_dir, 'tmp')
    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        scan_dirs.append(tmp_dir)

    for s_dir in scan_dirs:
        for item in os.listdir(s_dir):
            sub_dir = os.path.join(s_dir, item)
            if not os.path.isdir(sub_dir) or item in ['images', 'tmp', 'package', '__pycache__']:
                continue
            
            csv_path = os.path.join(sub_dir, 'arknights.csv')
            if not os.path.exists(csv_path):
                continue

            try:
                current_header, data, encoding = read_csv_data(csv_path)
                if current_header is None:
                    continue

                if current_header != expected_header:
                    print(f"跳过目录 {sub_dir}: arknights.csv 的表头不符合预期格式")
                    continue
                
                # 合并图片
                src_images_dir = os.path.join(sub_dir, 'images')
                if os.path.exists(src_images_dir) and os.path.isdir(src_images_dir):
                    for img_file in os.listdir(src_images_dir):
                        src_img = os.path.join(src_images_dir, img_file)
                        dst_img = os.path.join(target_images_dir, img_file)
                        if os.path.isfile(src_img):
                            shutil.copy2(src_img, dst_img)
                
                merged_data.extend(data)
                print(f"成功合并目录: {sub_dir} (编码: {encoding})")
            except Exception as e:
                print(f"处理目录 {sub_dir} 时出错: {e}")

    if merged_data:
        with open(target_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)
            writer.writerows(merged_data)
        print(f"\n合并完成，共 {len(merged_data)} 条记录 -> {target_csv_path}")
    else:
        print("\n未找到有效的 CSV 数据。")

if __name__ == '__main__':
    main()
