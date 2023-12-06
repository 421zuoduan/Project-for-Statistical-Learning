import os
from PIL import Image

# 定义文件夹路径
base_dir = './datasets'
norain_dir = os.path.join(base_dir, 'norain')
rainy_dir = os.path.join(base_dir, 'rain')
output_dir = os.path.join(base_dir, 'Rain100H/train_c')

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 循环处理每张图片
for i in range(1, 201):
    # norain_filename = f'norain-{i:03d}.png'
    # rainy_filename = f'rain-{i:03d}.png'
    # output_filename = f'norain-{i:03d}.png'  # 修改新图片的名称
    norain_filename = f'norain-{i}.png'
    rainy_filename = f'norain-{i}x2.png'
    output_filename = f'norain-{i}.png'  # 修改新图片的名称

    norain_path = os.path.join(norain_dir, norain_filename)
    rainy_path = os.path.join(rainy_dir, rainy_filename)
    output_path = os.path.join(output_dir, output_filename)

    # 打开两张图片
    norain_image = Image.open(norain_path)
    rainy_image = Image.open(rainy_path)

    # 获取图片尺寸
    width, height = norain_image.size

    # 创建新图片，尺寸是原图片的两倍宽度
    combined_image = Image.new('RGB', (width * 2, height))

    # 将norain图片粘贴到左半边，rainy图片粘贴到右半边
    combined_image.paste(norain_image, (0, 0))
    combined_image.paste(rainy_image, (width, 0))

    # 保存新图片
    combined_image.save(output_path)

    # 关闭图片文件
    norain_image.close()
    rainy_image.close()

print("拼接完成并保存到 'train_c' 文件夹内。")
