import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


BASE_PATH = Path(r"C:\Users\muham\OneDrive - TU Eindhoven\Extended-evaluation-snowpole-lidar-dataset\SnowPole_Detection_Dataset")
base_range = Path(r"C:\Users\muham\OneDrive - TU Eindhoven\Extended-evaluation-snowpole-lidar-dataset")
MODALITY_PATHS = {
    "nearir": BASE_PATH / "nearir",
    "signal": BASE_PATH / "signal",
    "reflec": BASE_PATH / "reflec",
    "range": base_range / "range-normalized",
}

CHANNEL_SELECTORS = {
    "nearir": lambda img: img[..., 2],
    "signal": lambda img: img[..., 0],
    "reflec": lambda img: img[..., 1],
    "range": lambda img: img,
}

SPLITS = ["train", "valid", "test"]

# (output_folder_name, (R, G, B))
CHANNEL_COMBINATIONS = [
    ("Combination4_nearir_signal_range", ("nearir", "signal", "range")),
    # ("Combination4_nearir_range_reflec", ("nearir", "range", "reflec")),
    # ("Combination4_range_signal_reflec", ("range", "signal", "reflec")),
    # ("Combination5_range_reflec_nearir", ("range", "reflec", "nearir")),
    # ("Combination5_signal_range_nearir", ("signal", "range", "nearir")),
    # ("Combination5_signal_reflec_range", ("signal", "reflec", "range")),
]

def collect_filenames(split, required_modalities):
    filename_sets = []
    for modality in required_modalities:
        split_dir = MODALITY_PATHS[modality] / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        filenames = {path.name for path in split_dir.glob("*.png")}
        if not filenames:
            raise FileNotFoundError(f"No PNG images found in {split_dir}")
        filename_sets.append(filenames)

    common_filenames = filename_sets[0]
    for other_set in filename_sets[1:]:
        common_filenames &= other_set

    if not common_filenames:
        raise ValueError(
            f"No common filenames across modalities {required_modalities} for split '{split}'."
        )

    return sorted(common_filenames)


def load_channel_image(modality, split, filename):
    image_path = MODALITY_PATHS[modality] / split / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    selector = CHANNEL_SELECTORS.get(modality)
    if selector is None:
        raise KeyError(f"No channel selector defined for modality '{modality}'")

    channel = selector(image)

    if channel.dtype == np.uint16:
        channel = cv2.convertScaleAbs(channel, alpha=255.0 / 65535.0)

    return channel


def generate_combination(combo_name, channel_modalities):
    required_modalities = set(channel_modalities)

    output_root = BASE_PATH / combo_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating {combo_name} ===")
    print(f"  Channels (R,G,B): {channel_modalities}")

    total_processed = 0

    for split in SPLITS:
        filenames = collect_filenames(split, required_modalities)
        output_split = output_root / split
        output_split.mkdir(parents=True, exist_ok=True)

        print(f"  Split '{split}': {len(filenames)} images")
        for filename in tqdm(filenames, desc=f"  {split}", leave=False):
            r_mod, g_mod, b_mod = channel_modalities
            r_img = load_channel_image(r_mod, split, filename)
            g_img = load_channel_image(g_mod, split, filename)
            b_img = load_channel_image(b_mod, split, filename)

            fused = np.stack([b_img, g_img, r_img], axis=-1)
            cv2.imwrite(str(output_split / filename), fused)
            total_processed += 1

        print(f"    ✓ Saved images to {output_split}")

    print(f"  Total images written for {combo_name}: {total_processed}")

for combo_name, channel_modalities in CHANNEL_COMBINATIONS:
    generate_combination(combo_name, channel_modalities)