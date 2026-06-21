import os
import re
from PIL import Image
from scipy.spatial import KDTree

# Extended list of base colors
COLORS_MAP = {
    "Red": (255, 0, 0),
    "Green": (0, 128, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Purple": (128, 0, 128),
    "Pink": (255, 192, 203),
    "Brown": (165, 42, 42),
    "Gray": (128, 128, 128),
    "Cyan": (0, 255, 255),
}


def get_actual_object_color(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            pixels = img.getdata()
            rgbs = []
            for r, g, b, a in pixels:
                if a > 100:
                    if not (r > 240 and g > 240 and b > 240) and not (r < 15 and g < 15 and b < 15):
                        rgbs.append((r, g, b))

            if not rgbs:
                return "Black" if (0, 0, 0) in pixels else "White"


            avg_r = sum(p[0] for p in rgbs) // len(rgbs)
            avg_g = sum(p[1] for p in rgbs) // len(rgbs)
            avg_b = sum(p[2] for p in rgbs) // len(rgbs)


            names = list(COLORS_MAP.keys())
            values = list(COLORS_MAP.values())
            kdtree = KDTree(values)
            _, index = kdtree.query((avg_r, avg_g, avg_b))
            return names[index]
    except Exception as e:
        print(f"Error in {image_path}: {e}")
        return None


def rename_fix(root_folder):
    subfolders = ['Neutral', 'Negative', 'Positive']

    color_names_pattern = "|".join(list(COLORS_MAP.keys()) + ["Black", "White"])

    for sub in subfolders:
        folder_path = os.path.join(root_folder, sub)
        if not os.path.exists(folder_path): continue

        print(f"Processing {sub}...")
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            file_path = os.path.join(folder_path, filename)
            name, ext = os.path.splitext(filename)

            clean_name = re.sub(r'_([0-9a-fA-F]{6})$', '', name)
            clean_name = re.sub(r'_(' + color_names_pattern + r')$', '', clean_name, flags=re.IGNORECASE)

            color_name = get_actual_object_color(file_path)

            if color_name:
                new_filename = f"{clean_name}_{color_name}{ext}"
                new_file_path = os.path.join(folder_path, new_filename)

                if filename != new_filename:
                    os.rename(file_path, new_file_path)
                    print(f"Fixed: {filename} -> {new_filename}")


if __name__ == "__main__":
    base_path = "OASIS_database_2016/final_standardized_stimuli"
    rename_fix(base_path)
    print("Done! Orange should now be labeled Orange or Red.")