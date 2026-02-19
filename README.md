# regressor-guided-image-editing
This is the code repository for the paper Regressor-Guided Generative Image Editing Balances User Emotions to Reduce Time Spent Online (https://arxiv.org/abs/2501.12289)

The repository builds upon imaginaire: https://github.com/NVlabs/imaginaire

## Setup

1. Clone the repository:
   - ``git clone https://github.com/christophgebhardt/regressor-guided-image-editing.git``


2. download COCO data from "https://cocodataset.org/#download":
    - `2017 Val images`
    - `2017 Train/Val annotations`

    Use this folder structure:
    ```
    [COCO_DIR]  
    ├── annotations  
    │   └── captions_val2017.json  
    └── val2017   
        └── *.jpg  
    ```
    Make sure ``COCO_DIR`` in ``paths.py`` points to this folder.

3. Install `uv`
    - ``curl -LsSf https://astral.sh/uv/install.sh | sh``

4) Run scripts from the src folder
- ``./[script_name.py]``