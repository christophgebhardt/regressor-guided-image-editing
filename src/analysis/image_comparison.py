import math
import os
import shutil
import json
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from analysis.predict_valence_arousal import predict_valence_arousal


def load_image_paths_from_json(json_path):
    """Loads image paths from a JSON file."""
    with open(json_path, 'r') as file:
        data = json.load(file)

    image_paths = []
    for image_item in data:
        image_paths.append(image_item['relative_path'])

    return image_paths


def get_files_from_subdirectories(base_directory):
    # List of supported image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Initialize an empty list to store image file paths
    image_files = []

    # Recursively go through all subdirectories and add image files to the list
    for dirpath, _, filenames in os.walk(base_directory):
        for filename in filenames:
            # Check if the file has an image extension
            if os.path.splitext(filename)[1].lower() in image_extensions:
                # Add the full path of the image file to the list
                file_path = os.path.join(
                    dirpath, filename).replace(base_directory + "/", '').replace(base_directory + "\\", '')
                image_files.append(file_path)

    return image_files


def find_images(base_dirs, relative_paths, file_contents=None):
    """Retrieves image paths for each relative path under each base directory."""
    image_files = {}
    for rel_path in relative_paths:
        if file_contents is not None and rel_path.split("/")[-1].replace(".jpg", "") not in file_contents:
            continue
        image_paths = []
        for base_dir in base_dirs:
            abs_path = os.path.join(base_dir, rel_path)
            if os.path.exists(abs_path):
                image_paths.append(abs_path)
        if len(image_paths) == len(base_dirs):
            image_files[rel_path] = image_paths
    return image_files


def show_images_in_subplot(image_paths):
    """Displays images in a single window with subplots for user comparison."""
    num_images = len(image_paths)
    num_rows = 2 if num_images >= 6 else 1
    num_cols = num_images if num_rows == 1 else (num_images + 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows + 0.3))

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_path = img_path.replace("\\", "/")
        cmap = None
        if i < len(image_paths) - 1 and img_path == image_paths[i + 1]:
            img = Image.open(img_path).convert("L")
            cmap = "gray"

        if num_rows > 1:
            ix_rows = 1 if i >= num_cols else 0
            ix_cols = i - ix_rows * num_cols
            ax = axes[ix_rows, ix_cols]
        else:
            ax = axes[i]

        ax.imshow(img, cmap=cmap)
        if "default" in img_path or "val2017" in img_path: # or "unadjusted" in img_path:
            title = "original"
            if i < len(image_paths) - 1 and img_path == image_paths[i + 1]:
                title = "greyscale"
        # elif "pos" in img_path:
        #     title = "pos"
        # elif "neutral" in img_path:
        #     title = "neutral"
        # elif "neg" in img_path:
        #     title = "neg"
        # elif "param" in img_path or "NAPS_PARAM_0.15" in img_path:
        #     title = "parametric optimization"
        # elif "diffusion" in img_path or "NAPS_XL_CG_CFG_2_0.20" in img_path:
        #     title = "diffusion"
        # elif "gan" in img_path or "NAPS_GAN_0.10" in img_path:
        #     title = "style optimization"
        else:
            # title = img_path.split("/")[-2]
            # weight = "s = " + img_path.split("/")[-2].split("_")[-1]
            # weight = "$w_2$ = " + img_path.split("/")[-2].split("_")[-1]
            try:
                ref = float(img_path.split("/")[-2].split("_")[-1])
            except ValueError:
                ref = None

            if "pos" in img_path:
                ref = "+" + str(ref/10) if ref is not None else 1.0
            elif "neg" in img_path:
                ref = "-" + str(ref/10) if ref is not None else -1.0
            elif "unadjusted" in img_path:
                ref = "original"

            weight = ref
            valence, arousal = predict_valence_arousal(img)
            va = f"({valence:.2f}, {arousal:.2f})"

            # color = 'black'
            color = 'white'
            ax.text(0.7, 0.01, weight, transform=ax.transAxes,
                    fontsize=28, color=color, ha='center', va='bottom')

            ax.text(0.65, 0.99, va, transform=ax.transAxes,
                    fontsize=28, color=color, ha='center', va='top')

        # ax.set_title(f"{i} - {title}")
        # ax.set_title(f"{title}")
        ax.axis("off")

    for ax in axes.flatten():
        # Check if data exists; if not, hide the axis
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'D:/Tmp/DIFF_{img_path.split("/")[-1]}')
    plt.show()


def user_select_image(image_paths, rel_path, destination_base, df=None):
    global df_bad, df_selected
    """Prompts the user to select the preferred image or skip."""
    show_images_in_subplot(image_paths)

    print(f"Select the preferred image:")
    for i, path in enumerate(image_paths):
        print(f"[{i}] - {path}")
    print("[-1] - Skip and proceed to the next image")

    if df is not None:
        df_print = df.drop(columns=['relative_path'])
        # the following two lines sort the stats of the image that is closest to the optimization objectives
        # Compute the Euclidean distance
        df_print['distance'] = df_print.apply(
            lambda row: math.sqrt(row['arousal'] ** 2 + (row['valence'] - 0.5) ** 2), axis=1)
        # Sort by the computed distance
        df_print = df_print.sort_values(by='distance', ascending=True).reset_index(drop=True)
        print(df_print)

    # choice = int(input("Enter the index of the preferred image (or -1 to skip): "))
    choice = 2

    selected_image = image_paths[choice]
    selected_image = selected_image.replace("\\", "/")
    image_name = selected_image.split("/")[-1]
    if choice == -1: # or "default" in selected_image:
        print(f"Skipping {rel_path}")
        os.remove(f'D:/Tmp/{image_name}')
        if df is not None:
            df = df[df["option"] == "original"]
            df_bad = pd.concat([df_bad, df]) if df_bad is not None else df
            df_bad.to_csv("bad_images.csv", index=False)
        return

    if destination_base is not None:
        dest_path = os.path.join(destination_base, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(selected_image, dest_path)
        print(f"Copied {selected_image} to {dest_path}")

    if df is not None:
        df = df[df["option"] == choice]
        is_different = int(input("Enter if the image is a highlight results (2), different from the original (1) or not (0): "))

        if is_different == 2:
            add_line_to_file(selected_image.split("/")[-1].replace(".jpg", ""))
            is_different = 1

        # Catch the warning
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")  # Ensure all warnings are caught
            df["is_different"] = is_different
        df_selected = pd.concat([df_selected, df]) if df_selected is not None else df
        df_selected.to_csv("selected_images.csv", index=False)
    elif choice != -1:
        # add_line_to_file(image_name.replace(".jpg", ""))
        pass


def main(json_path, base_dirs, destination_base, ref_file=None):
    # Load relative image paths from JSON
    if json_path is not None:
        relative_paths = load_image_paths_from_json(json_path)
    else:
        relative_paths = get_files_from_subdirectories(base_dirs[0])

    # Prompt for start index
    start_index = int(input(f"Enter the starting index (0 - {len(relative_paths) - 1}): "))

    file_contents = None
    if ref_file is not None:
        with open(ref_file, 'r') as file:
            file_contents = file.read()
        file_contents = file_contents.split("\n")

    # Find matching images for paths in JSON file
    image_files = find_images(base_dirs, relative_paths, file_contents)
    df = get_dataframes(base_dirs) if json_path is not None else None
    # stats_df = compute_grouped_statistics(df, group_by_column="option", exclude_columns=["relative_path"])
    # stats_df.to_csv("image_stats.csv")

    # Iterate through images starting from the specified index
    ix = start_index
    for rel_path in list(image_files.keys())[start_index:]:
        print(f"[{ix}/{len(relative_paths)}] - {rel_path}")
        tdf = df[df["relative_path"] == rel_path.split("/")[-1]] if df is not None else None
        user_select_image(image_files[rel_path], rel_path, destination_base, tdf)
        ix += 1


def get_dataframes(base_dirs):
    """Retrieves image paths for each relative path under each base directory."""
    original_index = next((i for i, s in enumerate(base_dirs) if "default" in s), None)
    original_df = pd.read_json(f'{base_dirs[original_index]}/feed_images.json')
    original_df = original_df.drop('captions', axis=1)
    original_df["delta_valence"] = 0.0
    original_df["delta_arousal"] = 0.0
    original_df["rec_error"] = 0.0
    original_df.insert(0, 'option', "original")
    original_df['relative_path'] = original_df['relative_path'].apply(rename_relative_path_value)

    df = original_df
    for i, img_dir in enumerate(base_dirs):
        tdf = None
        if os.path.exists(f'{img_dir}/stats.csv'):
            tdf = pd.read_csv(f'{img_dir}/stats.csv')

        if tdf is not None:
            tdf.insert(0, 'option', i)
            tdf['relative_path'] = tdf['relative_path'].apply(rename_relative_path_value)
            if "delta_valence" not in tdf.columns:
                tdf = tdf.sort_values(by="relative_path").reset_index(drop=True)
                original_df = original_df.sort_values(by="relative_path").reset_index(drop=True)

                tdf["delta_valence"] = tdf["valence"] - original_df["valence"]
                tdf["delta_arousal"] = tdf["arousal"] - original_df["arousal"]

            df = pd.concat([df, tdf], ignore_index=True)

    return df


def rename_relative_path_value(value):
    return value.split("/")[-1]


def compute_grouped_statistics(df, group_by_column="option", exclude_columns=None):
    """
    Compute the mean and standard deviation for all columns in the DataFrame,
    grouped by the specified column (default is 'option'), excluding specified columns.

    Parameters:
    - df: DataFrame, contains the data
    - group_by_column: str, the column name to group by (default is 'option')
    - exclude_columns: list, columns to be excluded from the calculations (default is None)

    Returns:
    - stats_df: DataFrame, containing the mean and standard deviation for each group
    """
    if exclude_columns is None:
        exclude_columns = []

    # Select columns to consider
    columns_to_consider = [col for col in df.columns if col not in exclude_columns and col != group_by_column]

    # Compute the mean for each group
    mean_df = df.groupby(group_by_column)[columns_to_consider].mean().add_suffix('_mean')

    # Compute the standard deviation for each group
    std_df = df.groupby(group_by_column)[columns_to_consider].std().add_suffix('_std')

    # Merge the mean and standard deviation DataFrames
    stats_df = mean_df.merge(std_df, left_index=True, right_index=True)

    # Print the computed statistics
    print("Grouped Statistics (Mean and Standard Deviation):")
    print(stats_df)

    return stats_df


def add_line_to_file(line, file_path="highlight_results.txt"):
    """Appends a line to a text file."""
    with open(file_path, 'a') as file:
        file.write(line + '\n')


# Global dataframes
df_bad = pd.read_csv("bad_images.csv") if os.path.exists("bad_images.csv") else None
df_selected = pd.read_csv("selected_images.csv") if os.path.exists("selected_images.csv") else None

# Selected for param: param_2024_11_04_08_36_02; Selected for gan: gan_s5
# json_path = "../caption_files/feed_images.json"
# directory1 = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/Instagram/default"
# base_directory = "/media/chris/Elements/Projects/Generative Emotional Image Adaptations/Study 2 Images"
# base_dirs = [d.path for d in os.scandir(base_directory) if d.is_dir() and "diffusion" in d.path]
# destination_base = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/Instagram/diffusion"

# json_path = None
# directory1 = "D:/GitRepos/emotion-adaptation/coco/val2017"
# base_directory = "D:/Projects/Generative Emotional Image Adaptations/Results/COCO_SD"
# # base_directory = "D:/Projects/Generative Emotional Image Adaptations/Results/COCO_GAN"
# base_dirs = [d.path for d in os.scandir(base_directory) if d.is_dir() and "CG" in d.path and "XL" not in d.path and "CFG" not in d.path]
# # base_dirs = [d.path.replace("\\", "/") for d in os.scandir(base_directory) if d.is_dir()]
# # base_dirs.append(directory1)
# destination_base = None
# ref_file = "../feed_results/highlight_results_sd_coco.txt"

# json_path = None
# base_directory = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/Instagram"
# base_dirs = [f"{base_directory}/default", f"{base_directory}/default", f"{base_directory}/param", f"{base_directory}/diffusion"]
# destination_base = None
# ref_file = "../feed_results/highlight_results.txt"

# json_path = None
# base_directory = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/NAPS"
# base_dirs = [f"{base_directory}/default", f"{base_directory}/default", f"{base_directory}/NAPS_PARAM_0.15",
#              f"{base_directory}/NAPS_GAN_0.10", f"{base_directory}/NAPS_XL_CG_CFG_2_0.20"]
# destination_base = None
# ref_file = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Backend/db/stimuli/fixed_imgs_within.txt"

json_path = None
directory1 = "D:/Projects/Generative Emotional Image Adaptations/Results/relative_change/unadjusted"
base_directory = "D:/Projects/Generative Emotional Image Adaptations/Results/relative_change/diffusion"
base_dirs = [d.path.replace("\\", "/") for d in os.scandir(base_directory) if d.is_dir() and "neutral" not in d.path]
# base_directory = "D:/Projects/Generative Emotional Image Adaptations/Results/relative_change/param"
# sub_dirs = ["neg_02", "neg_01", "pos_01", "pos_02"]
# base_dirs = [base_directory + "/" + d for d in sub_dirs]
base_dirs.insert(int(len(base_dirs) / 2), directory1)
destination_base = None
ref_file = "../feed_results/highlight_results_relative.txt"
# ref_file = None

main(json_path, base_dirs, destination_base, ref_file)
