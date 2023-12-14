# AML_P3
AML Course Task 3 2023


Preprocessing ideas:

- Resizing
- Denoising: NLM denoising ? DRUnet as seen in other project ?
- Normalization
- Image Histogram equlization
- Data Augmentation

Models:

- MaskDINO
- Classic Unet ?

## MaskDINO information

### Installation

Checkout the `INSTALL.md` and `maskDINO_README.md` for installation istructions.

### Data

The dataset was transfomred to COCO format using `viewdata_and_convert_to_coco.py`, but the ready data is already in the `datasets/coco` folder.

### Training

The model can be trained by running:

```bash
python train_net.py --config-file best_model_config.yaml
```

The `best_model_config.yaml` is a config of the best model we trained. Before training don't forget to change output directory in the config file, as well as the starting weights path.
