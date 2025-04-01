import json

# 从二进制文件读取 JSON 字符串
with open('mkm.bin', 'rb') as f:
    json_str = f.read().decode('utf-8')

# 解析 JSON 数据
results = json.loads(json_str)

# 打印结果
print(results)