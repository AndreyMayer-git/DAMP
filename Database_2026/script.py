import os
import pandas as pd
import numpy as np
import random
from rembg import remove
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

CSV_FILE = 'OASIS.csv'
INPUT_DIR = 'images'
OUTPUT_DIR = 'processed_stimuli'

TARGET_SIZE = 700
OBJECT_FILL = 0.75

TH_NEGATIVE = 3.0
TH_POSITIVE = 5.0
BLUR_RADIUS = 8
NOISE_FACTOR = 20
BRIGHTNESS_FACTOR = 2.0  # lower further if the color wash overwhelms the object's form (curation criterion 1)

CATEGORIES = ['Positive', 'Negative', 'Neutral']

# Path A: colour is CROSSED with valence, not determined by it. A single shared palette is
# used for every category so that hue and luminance cannot proxy for affective category.
# Colours are assigned in a balanced round-robin within each category (see main()), making
# colour orthogonal to valence -> this both removes the colour/brightness confound and
# enables the intended dissociation ("avoids red specifically" vs "avoids negative in general").
# For rigorous low-level equalisation (luminance, contrast, spatial frequency) the standard
# tool is the SHINE toolbox (Willenbockel et al., 2010, Behavior Research Methods); the palette
# below is chosen to be roughly matched in lightness as a first approximation.
SHARED_PALETTE = ['#C85C5C', '#5C8FC8', '#5FB05F', '#C8A23C', '#9B6FC0', '#46AEAE', '#CE7FB0', '#8C8F98']



def process_single_image(image_path, save_path, target_color):
    try:
        img = Image.open(image_path).convert("RGBA")
        img_no_bg = remove(img)

        bbox = img_no_bg.getbbox()
        if not bbox:
            return False

        obj_cropped = img_no_bg.crop(bbox)

        alpha = obj_cropped.split()[3]
        gray = obj_cropped.convert("L")
        gray = ImageOps.autocontrast(gray, ignore=[0])

        enhancer = ImageEnhance.Brightness(gray)
        gray = enhancer.enhance(BRIGHTNESS_FACTOR)

        blurred = gray.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
        noisy = add_noise(blurred, NOISE_FACTOR)

        colorize = ImageOps.colorize(noisy, black="black", white=target_color)
        final_obj = colorize.convert("RGBA")
        final_obj.putalpha(alpha)

        canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))

        w, h = final_obj.size
        max_dim = max(w, h)
        ratio = (TARGET_SIZE * OBJECT_FILL) / max_dim
        new_size = (int(w * ratio), int(h * ratio))

        final_obj = final_obj.resize(new_size, Image.Resampling.LANCZOS)

        offset = ((TARGET_SIZE - final_obj.size[0]) // 2,
                  (TARGET_SIZE - final_obj.size[1]) // 2)

        canvas.paste(final_obj, offset, final_obj)

        canvas.save(save_path, "PNG")
        return True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def add_noise(image, factor):
    img_array = np.array(image)
    gauss = np.random.normal(0, factor ** 0.5, img_array.shape)
    noisy_array = np.clip(img_array + gauss, 0, 255).astype('uint8')
    return Image.fromarray(noisy_array)


def setup_directories():
    for cat in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)


def main():
    setup_directories()
    df = pd.read_csv(CSV_FILE)
    processed_count = 0

    # Balanced colour assignment: each category draws the shared palette in a shuffled
    # round-robin, so every colour appears about equally often in every valence category

    color_cycles = {cat: random.sample(SHARED_PALETTE, len(SHARED_PALETTE)) for cat in CATEGORIES}
    color_idx = {cat: 0 for cat in CATEGORIES}

    for _, row in df.iterrows():
        theme = row['Theme'].strip()
        valence = row['Valence_mean']

        filename = f"{theme}.jpg"
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path):
            input_path = os.path.join(INPUT_DIR, f"{theme}.jpeg")

        if not os.path.exists(input_path):
            continue

        category = 'Positive' if valence > TH_POSITIVE else ('Negative' if valence < TH_NEGATIVE else 'Neutral')
        cyc = color_cycles[category]
        color_hex = cyc[color_idx[category] % len(cyc)]
        color_idx[category] += 1

        save_path = os.path.join(OUTPUT_DIR, category, f"{theme}_centered.png")

        if process_single_image(input_path, save_path, color_hex):
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed: {processed_count}")

    print(f"Done! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()