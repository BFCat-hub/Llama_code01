import os 
import json 
import re 
  
# 文件夹路径  
c_folder = "./cpp.para.test"
cuda_folder = "./cuda.para.test"

#c_folder = "./cpp.para.valid"
#cuda_folder = "./cuda.para.valid"

def remove_comments(content):
    # 正则表达式匹配多行注释和单行注释
    multi_line_comment_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)
    single_line_comment_pattern = re.compile(r'//.*?(?=\n|$)', re.MULTILINE)
    # 移除多行注释
    content = re.sub(multi_line_comment_pattern, '', content)
    # 移除单行注释
    content = re.sub(single_line_comment_pattern, '', content)
    return content

  
# 确保文件夹存在  
if not os.path.exists(c_folder) or not os.path.exists(cuda_folder):  
    print("Error: One or both folders do not exist.")  
    exit(1)  
  
# 获取文件夹中文件列表  
c_files = os.listdir(c_folder)  
cuda_files = os.listdir(cuda_folder)  
  
# 确保两个文件夹中的文件数量相同  
if len(c_files) != len(cuda_files):  
    print("Error: The number of files in the serial and CUDA folders is not the same.")  
    exit(1)  
  
# 创建一个列表来保存JSON对象  
programs = []  



  
# 遍历文件，创建JSON对象  
for c_file, cuda_file in zip(c_files, cuda_files):  
    # 确保文件名相同（除了扩展名）  
    if os.path.splitext(c_file)[0].split("_")[1] != os.path.splitext(cuda_file)[0].split("_")[1]:  
        print(f"Error: Mismatched file names: {c_file} and {cuda_file}")  
        continue  
  
    # 读取文件内容  
    with open(os.path.join(c_folder, c_file), 'r', encoding='utf-8') as f:  
        c_code = f.read()  
        #去除注释
        c_code = remove_comments(c_code)
        #去除换行符
        #c_code = c_code.replace("\n"," ")
        
  
    with open(os.path.join(cuda_folder, cuda_file), 'r', encoding='utf-8') as f:  
        mpi_code = f.read()  
        #去除注释
        mpi_code = remove_comments(mpi_code)
        #去除换行符
        #mpi_code = mpi_code.replace("\n"," ")
        
  
    # 创建JSON对象  
    program = {   
        "id": os.path.splitext(c_file)[0].split("_")[1],  
        "c_code": c_code,  
        "cuda_code": mpi_code
    }  
  
    # 添加到列表中  
    programs.append(program)  
  
# 将列表保存为JSON文件  
with open('test02.json', 'w') as f:
    json.dump(programs, f, indent=4)  
  
print("JSON file created successfully.")