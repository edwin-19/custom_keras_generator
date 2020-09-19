# Model Conversion and Custom generator
This is a repo to test out a few things some of which are:
1) custom data generators with augments
2) resnet vs effnet
3) model conversion from onnx to tensorrt

## Dataset Used
Intel Image Classification you can download it [here](https://www.kaggle.com/puneet6060/intel-image-classification)

## Model
You can download the efficientnet models [here](https://drive.google.com/drive/folders/1a20ZPQjbipcJp33Q3JjELWqn2dgPthKA?usp=sharing)

## Inferece
```sh
python inference.py -i data/image.jpg -w weights/effnet_b5_weights.h5
```

## Notes:
Checkout training in the notebooks section