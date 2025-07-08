# gbk_to_utf8.py
input_file = 'all.csv'
output_file = 'all_utf8.csv'

with open(input_file, 'r', encoding='gbk') as f_in:
    content = f_in.read()

with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write(content)

print("转换完成，已保存为 all_utf8.csv")

