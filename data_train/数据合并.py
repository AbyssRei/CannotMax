import sys
import zipfile
import csv
import io
import shutil
from pathlib import Path

# 获取项目根目录
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent

# 将项目根目录添加到 sys.path 以便导入 config
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import MONSTER_COUNT, FIELD_FEATURE_COUNT


def get_expected_header():
    """根据配置生成预期表头"""
    if FIELD_FEATURE_COUNT > 0:
        header = [f"{i + 1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}LF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += [f"{i + 1}R" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}RF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += ["Result", "ImgPath"]
    else:
        header = [f"{i + 1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}R" for i in range(MONSTER_COUNT)]
        header += ["Result", "ImgPath"]
    return header


def read_csv_from_zip(zip_ref, csv_filename):
    """从 ZIP 文件流中直接读取 CSV 数据，尝试多种编码"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']

    for encoding in encodings:
        try:
            with zip_ref.open(csv_filename) as f:
                text_f = io.TextIOWrapper(f, encoding=encoding, newline='')
                reader = csv.reader(text_f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, [], encoding
                data = list(reader)
                return header, data, encoding
        except (UnicodeDecodeError, io.UnsupportedOperation):
            continue

    raise ValueError(f"无法以支持的编码读取压缩包内的文件 {csv_filename}")


def process_archives(merge_images=True, extract_result_images=False):
    package_dir = base_dir / "package"
    target_images_dir = base_dir / 'images'
    target_csv_path = base_dir / 'arknights.csv'

    # 1. 初始化和清理环境
    if not package_dir.exists():
        print(f"未找到压缩包目录: {package_dir}")
        return

    if merge_images:
        if target_images_dir.exists():
            print(f"正在清空旧图片目录: {target_images_dir}")
            shutil.rmtree(target_images_dir)
        target_images_dir.mkdir(parents=True, exist_ok=True)

    target_csv_path.unlink(missing_ok=True)

    expected_header = get_expected_header()
    img_path_idx = expected_header.index("ImgPath")

    # 内存变量，存放所有合并后的数据行
    merged_data = []
    seen_img_paths = set()

    # 定义可能的文件后缀，涵盖常见大小写
    possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

    zip_files = list(package_dir.glob("*.zip"))
    print(f"找到 {len(zip_files)} 个压缩包，准备提取并合并...")

    # 2. 直接遍历压缩包
    for zip_path in zip_files:
        print(f"\n正在处理压缩包: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 建立 namelist 的 O(1) 查找集合
                zip_namelist_set = set(zf.namelist())

                csv_members = [m for m in zip_namelist_set if m.endswith('arknights.csv')]

                for csv_member in csv_members:
                    prefix = csv_member.rsplit('arknights.csv', 1)[0]
                    header, data, encoding = read_csv_from_zip(zf, csv_member)

                    if header is None or header != expected_header:
                        print(f"  [跳过] {csv_member}: 表头不符合预期格式")
                        continue

                    added_count = 0
                    skip_count = 0

                    for row in data:
                        img_path = row[img_path_idx]

                        if img_path not in seen_img_paths:
                            merged_data.append(row)
                            seen_img_paths.add(img_path)
                            added_count += 1

                            if merge_images:
                                # 表格里没有后缀，只取基础文件名
                                img_filename_base = Path(img_path).name

                                # 根据参数决定需要提取哪些后缀形式的文件
                                suffixes_to_extract = [""]
                                if extract_result_images:
                                    suffixes_to_extract.append("_result")

                                for suffix in suffixes_to_extract:
                                    zip_img_path = None
                                    actual_ext = ""

                                    # 探测实际文件后缀
                                    for ext in possible_extensions:
                                        candidate_path = f"{prefix}images/{img_filename_base}{suffix}{ext}"
                                        if candidate_path in zip_namelist_set:
                                            zip_img_path = candidate_path
                                            actual_ext = ext
                                            break

                                    if zip_img_path:
                                        target_img_path = target_images_dir / f"{img_filename_base}{suffix}{actual_ext}"
                                        try:
                                            with zf.open(zip_img_path) as source_file:
                                                with open(target_img_path, 'wb') as target_file:
                                                    shutil.copyfileobj(source_file, target_file)
                                        except Exception as e:
                                            print(f"  [错误] 提取图片 {zip_img_path} 失败: {e}")
                                    else:
                                        # 如果是原图没找到报警告；如果是_result图没找到则静默跳过即可
                                        if suffix == "":
                                            print(
                                                f"  [警告] 找不到对应的图片: {prefix}images/{img_filename_base}[后缀]")

                        else:
                            skip_count += 1

                    print(
                        f"  -> 成功合并: {csv_member} (编码: {encoding}, 新增: {added_count}, 重复跳过: {skip_count})")

        except zipfile.BadZipFile:
            print(f"压缩包损坏，跳过: {zip_path.name}")
        except Exception as e:
            print(f"处理压缩包 {zip_path.name} 时出错: {e}")

    # 3. 写入最终的 CSV 文件
    if merged_data:
        print(f"\n开始写入最终的 CSV 文件...")
        with open(target_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)
            writer.writerows(merged_data)
        print(f"处理完成！共 {len(merged_data)} 条有效记录 -> {target_csv_path}")
    else:
        print("\n未找到有效的 CSV 数据。")


if __name__ == '__main__':
    merge_imgs = False # 设置为 True 则提取阵容图
    extract_res_imgs = False  # 设置为 True 则同时提取带有 _result 的结果图
    process_archives(merge_images=merge_imgs, extract_result_images=extract_res_imgs)
