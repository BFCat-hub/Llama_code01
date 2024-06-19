import os

# 必备的CUDA头文件列表
required_headers = [
    "#include <stdio.h>",
    "#include <device_launch_parameters.h>",
]

def check_and_add_headers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 检查现有的头文件
    existing_headers = [line.strip() for line in lines if line.strip().startswith("#include")]

    # 找出缺失的头文件
    missing_headers = [header for header in required_headers if header not in existing_headers]

    if missing_headers:
        # 将缺失的头文件添加到文件的开头
        new_lines = missing_headers + ["\n"] + lines
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)
        print(f"Added missing headers to {file_path}")
    else:
        print(f"All required headers are present in {file_path}")

def add_missing_headers_to_cuda_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cu"):
                file_path = os.path.join(root, file)
                check_and_add_headers(file_path)

# 指定包含CUDA文件的目录
cuda_files_directory = "cuda.para.test"

# 批量添加缺失的头文件
add_missing_headers_to_cuda_files(cuda_files_directory)
