import os
import csv
import shutil

def read_csv_data(filepath):
    """读取CSV文件，自动检测编码并返回表头和数据行（不去重）"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, [], encoding # 空文件返回空数据
                data = list(reader) # 直接转为列表，不去重
                return header, data, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法以支持的编码读取文件 {filepath}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_images_dir = os.path.join(base_dir, 'images')
    target_csv_path = os.path.join(base_dir, 'arknights.csv')

    # 创建目标图片文件夹
    os.makedirs(target_images_dir, exist_ok=True)

    merged_data = []
    header = None

    # 遍历当前文件夹下的所有子文件夹
    for item in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir, item)
        
        # 跳过目标images文件夹，以及非文件夹的文件
        if not os.path.isdir(sub_dir) or item == 'images':
            continue

        # 1. 复制子文件夹内的图片
        src_images_dir = os.path.join(sub_dir, 'images')
        if os.path.exists(src_images_dir) and os.path.isdir(src_images_dir):
            for img_file in os.listdir(src_images_dir):
                src_img = os.path.join(src_images_dir, img_file)
                dst_img = os.path.join(target_images_dir, img_file)
                if os.path.isfile(src_img):
                    shutil.copy2(src_img, dst_img)
            print(f"成功合并图片目录: {src_images_dir}")

        # 2. 合并子文件夹内的 csv 文件
        for file in os.listdir(sub_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(sub_dir, file)
                try:
                    current_header, data, encoding = read_csv_data(csv_path)
                    
                    if current_header is None:
                        continue # 忽略空文件

                    if header is None:
                        header = current_header
                    elif header != current_header:
                        print(f"警告：文件 {csv_path} 的表头与其他文件不一致，可能会导致数据错位")
                    
                    merged_data.extend(data)
                    print(f"成功读取并合并了文件: {csv_path} (编码: {encoding})")
                except Exception as e:
                    print(f"读取文件 {csv_path} 时出错: {e}")

    # 3. 写入合并后的 CSV
    if header is not None:
        with open(target_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(merged_data)
        print(f"\n所有CSV文件合并完成，共 {len(merged_data)} 条记录，结果已保存到: {target_csv_path}")
    else:
        print("\n未找到任何有效的 CSV 数据进行合并。")

    print(f"所有子文件夹中的图片已提取并合并至: {target_images_dir}")

if __name__ == '__main__':
    main()
