import zipfile
from pathlib import Path
import os

def unpackage():
    # 当前脚本所在目录
    current_dir = Path(__file__).parent
    # 压缩包所在目录
    package_dir = current_dir / "package"
    
    if not package_dir.exists():
        print(f"未找到目录: {package_dir}")
        return

    # 寻找所有的 .zip 文件
    zip_files = list(package_dir.glob("*.zip"))
    
    if not zip_files:
        print("未找到任何 .zip 文件")
        return

    print(f"找到 {len(zip_files)} 个压缩包，准备解压到: {current_dir}")

    for zip_path in zip_files:
        print(f"正在解压: {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(current_dir)
            print(f"成功解压: {zip_path.name}")
        except Exception as e:
            print(f"解压 {zip_path.name} 时出错: {e}")

if __name__ == "__main__":
    unpackage()
