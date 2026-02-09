from torchvision import transforms
from PIL import Image
from analysis.predict_valence_arousal import predict_valence_arousal
import matplotlib.pyplot as plt


def main():
    add_valence_arousal = True
    is_transform = False
    # color = 'black'
    color = 'white'

    output_size = 1024
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.CenterCrop(output_size)
    ])

    image_names = ["000000054931.jpg", "000000172877.jpg", "000000338625.jpg", "000000514376.jpg", "000000579655.jpg"]
    # base_directory = "D:/GitRepos/emotion-adaptation/coco/val2017"
    base_directory = "D:/Projects/Generative Emotional Image Adaptations/Results/COCO_SD/CFG_2_edit"
    base_directory_adapted = "D:/Projects/Generative Emotional Image Adaptations/Results/Highlights/COCO_CFG_EDIT_VA"
    flag = "CFG_"

    for image_name in image_names:
        image_path = f"{base_directory}/{image_name}"
        black_flag = "/black" if color == "black" and add_valence_arousal else ""
        img_adapted_path = f"{base_directory_adapted}{black_flag}/{flag}{image_name}"

        image = Image.open(image_path)
        if is_transform:
            image = transform(image)

        if not add_valence_arousal:
            image.save(img_adapted_path)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(image)

            valence, arousal = predict_valence_arousal(image)
            va = f"({valence:.2f}, {arousal:.2f})"

            ax.text(0.65, 0.99, va, transform=ax.transAxes,
                    fontsize=28, color=color, ha='center', va='top')

            ax.axis("off")
            plt.tight_layout()
            plt.savefig(img_adapted_path)
            plt.show()


if __name__ == "__main__":
    main()
