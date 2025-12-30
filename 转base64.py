import base64

# 将结果填入 url 字段内
def encode_image(image_path):
    extension = image_path.split(".")[-1]
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{extension};base64,{base64_image}"

print(encode_image("/Users/bytedance/PythonProgram/OpenManus/图片test1.jpeg"))
