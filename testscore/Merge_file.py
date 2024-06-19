import os
import re


def remove_comments(source):
    """
    移除源代码中的单行和多行注释
    """
    # 匹配多行注释和单行注释的正则表达式
    multi_line_comment = re.compile(r'/\*.*?\*/', re.DOTALL)
    single_line_comment = re.compile(r'//.*?\n')

    # 先去除多行注释
    source_no_multi_comments = re.sub(multi_line_comment, '', source)
    # 再去除单行注释
    source_no_comments = re.sub(single_line_comment, '', source_no_multi_comments)

    return source_no_comments


def read_files_and_merge(input_dir, output_file):
    # 获取输入目录下的所有文件名
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.cu')])

    with open(output_file, 'w') as outfile:
        for file in files:
            input_file_path = os.path.join(input_dir, file)
            with open(input_file_path, 'r') as infile:
                # 读取文件内容
                content = infile.read()
                # 去除注释
                content_no_comments = remove_comments(content)
                # 将内容转换为一行
                one_line_content = content_no_comments.replace('\n', ' ').replace('\r', ' ')
                outfile.write(one_line_content + '\n')


# 示例调用
input_directory = './refer_cuda01'  # CUDA 文件所在目录
output_filename = 'refer_cudamerge02.cu'  # 合并后的输出文件名

read_files_and_merge(input_directory, output_filename)
