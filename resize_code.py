import os
from PIL import Image
from tqdm import tqdm

def resize_images_in_folder(input_folder, output_folder, size):
    # 입력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더 내의 모든 파일에 대해 반복
    filenames = os.listdir(input_folder)
    for filename in tqdm(filenames, desc="Resizing images"):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            resize_image(input_image_path, output_image_path, size)

def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    aspect_ratio = width / height
    if width > height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)
    resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
    resized_image.save(output_image_path)

if __name__ == "__main__":
    input_folder = "/home/gyutae/atops2019/original_data/test_images"  # 입력 폴더 경로
    output_folder = "/home/gyutae/atops2019/resie_original_data/test"  # 출력 폴더 경로
    size = 512  # 새로운 이미지 크기

    resize_images_in_folder(input_folder, output_folder, size)
