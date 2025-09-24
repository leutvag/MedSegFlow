import os
from PIL import Image

def generate_symmetries(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            name, ext = os.path.splitext(filename)

            # 0°, 90°, 180°, 270°
            rotations = {
                "rot0": img,
                "rot90": img.rotate(90, expand=True),
                "rot180": img.rotate(180, expand=True),
                "rot270": img.rotate(270, expand=True),
            }

            for rot_name, rot_img in rotations.items():
                # Store Rotation
                rot_img.save(os.path.join(output_folder, f"{name}_{rot_name}{ext}"))

                # Flip 
                flipped = rot_img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped.save(os.path.join(output_folder, f"{name}_{rot_name}_flip{ext}"))

            print(f"Έγιναν όλες οι συμμετρίες για: {filename}")

if __name__ == "__main__":
    input_folder_yes = "[path2yes]/yes" # bale swsto path   
    output_folder_no = "symmetries_yes" 

    input_folder_no = "[path2yes]/no"  #bale swsto path  
    output_folder_no = "symmetries_no

    generate_symmetries(input_folder_yes, output_folder_yes)
    generate_symmetries(input_folder_no, output_folder_np)
