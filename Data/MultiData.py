import os

def copy_file_contents(input_file, output_file, num_copies):
    # 打开输入文件以读取内容
    with open(input_file, 'r') as file:
        contents = file.read()

    # 构建复制后的内容
    copied_contents = contents * num_copies

    # 打开输出文件以写入内容
    with open(output_file, 'w') as file:
        file.write(copied_contents)

# 输入文件路径
input_file = 'lol_corpus.txt'

# 输出文件路径
output_file = 'lol_corpus_100.txt'

# 复制的倍数
num_copies = 100

# 调用函数进行复制
copy_file_contents(input_file, output_file, num_copies)

