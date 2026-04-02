import os
import pandas as pd
import torch_fidelity

from classifier_score_of_images import get_classifier_score_of_images
from statistical_analysis import one_way_stats_test
import low_level_image_metrics as llim
from ListDataset import ListDataset


def main():
    # output_path = f"{directory}/NAPS_SD"
    output_path = f"/home/cgebhard/diffusion-guidance/NAPS_SD"
    process_result_images(output_path)


def process_result_images(folder_path):
    """
    Iterates through images in a folder, computes valence and arousal scores,
    and saves them in a dictionary where the flag is the key.
    """
    scores_dict = {"method": [], "valence": [], "arousal": [], "saturation": [], "bright": [], "colorful": [],
                   "light": [], "contrast": [], "blur": [], "image_path": []}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .jpg image with the required format
        if filename.endswith(".jpg") and "_" in filename:
            try:
                # Extract flag from the filename
                name, flag_with_ext = filename.rsplit("_", 1)
                flag = flag_with_ext.split(".")[0]
                if len(flag) == 1:
                    flag = "original"

                # Construct the full image path
                image_path = os.path.join(folder_path, filename)

                # Compute valence and arousal scores
                score = get_classifier_score_of_images([image_path], "va")
                valence = score[0, 0].item()
                arousal = score[0, 1].item()
                # print(f"Score {flag}: valence {score[0, 0].item()}, arousal {score[0, 1].item()}")

                # Append the scores to the dictionary
                scores_dict["method"].append(flag)
                scores_dict["valence"].append(valence)
                scores_dict["arousal"].append(arousal)
                scores_dict["saturation"].append(llim.compute_mean_saturation(image_path))
                scores_dict["bright"].append(llim.compute_mean_brightness(image_path))
                scores_dict["colorful"].append(llim.calculate_colorfulness(image_path))
                scores_dict["light"].append(llim.compute_lighting_diversity(image_path))
                scores_dict["contrast"].append(llim.compute_rms_contrast(image_path))
                scores_dict["blur"].append(llim.compute_blur_effect(image_path))
                scores_dict["image_path"].append(image_path)
                                              
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(scores_dict)
    one_way_stats_test(df, "method", "valence")
    one_way_stats_test(df, "method", "arousal")
    one_way_stats_test(df, "method", "saturation")
    one_way_stats_test(df, "method", "bright")
    one_way_stats_test(df, "method", "colorful")
    one_way_stats_test(df, "method", "light")
    one_way_stats_test(df, "method", "contrast")
    one_way_stats_test(df, "method", "blur")

    # Group 'imagepath' based on 'method'
    grouped_data = df.groupby('method')['image_path'].apply(list).to_dict()

    # Print the grouped data
    for method, image_paths in grouped_data.items():
        if method == "original":
            continue

        results_quality = torch_fidelity.calculate_metrics(
            input1=ListDataset("", grouped_data["original"]),
            input2=ListDataset("", image_paths),
            isc=True,
            fid=True,
            kid=True,
            cuda=True,  # Use GPU if available
            batch_size=20,  # Adjust batch size as needed
            kid_subset_size=20  # Necessary to set if running on small dataset
        )
        print(method)
        print(results_quality)


if __name__ == '__main__':
    main()
