import os
from PIL import Image

# 定义文件夹路径
base_dir = './datasets'
rainy_dir = os.path.join(base_dir, 'rainy')
output_dir = os.path.join(base_dir, 'testreal/test_real')

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 遍历并重命名文件
for i, file in enumerate(os.listdir(rainy_dir)):
    new_filename = f'rain-{i+1}.png'

    old_path = os.path.join(rainy_dir, file)
    new_path = os.path.join(rainy_dir, new_filename)

    os.rename(old_path, new_path)

contents = os.listdir(rainy_dir)

# 循环处理每张图片
for i in range(1, len(contents) + 1):
    rainy_filename = f'rain-{i}.png'
    output_filename = f'rain-{i}.png'  # 修改新图片的名称

    rainy_path = os.path.join(rainy_dir, rainy_filename)
    output_path = os.path.join(output_dir, output_filename)

    # 打开重命名后的文件
    rainy_image = Image.open(rainy_path)

    # 获取rainy图片尺寸
    width, height = rainy_image.size

    # 创建空白图片，与rainy图片大小相同
    norain_image = Image.new('RGB', (width, height))

    # 创建新图片，尺寸是原图片的两倍宽度
    combined_image = Image.new('RGB', (width * 2, height))

    # 将norain图片粘贴到左半边，rainy图片粘贴到右半边
    combined_image.paste(norain_image, (0, 0))
    combined_image.paste(rainy_image, (width, 0))

    # 保存新图片
    combined_image.save(output_path)

    # 关闭图片文件
    rainy_image.close()

print("拼接完成并保存到 'test_real' 文件夹内。")
