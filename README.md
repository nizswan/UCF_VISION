# UCF VISION
Publically Available Dataset(s): https://drive.google.com/drive/folders/1IUVSkOcDjggGhobZBP3_V69BegmhDnkp?usp=drive_link 

The true tree from the directory of training should look like:
.
├── run.slurm
└── src
    ├── data
    │   ├── k1
    │   ├── k2
    │   ├── k3
    │   └── k5
    ├── distance_matrix.py
    ├── models
    │   ├── generalclassifier.py
    │   ├── onnx_transfer.py
    │   ├── predictor.onnx
    │   ├── predictor.onnx.data
    │   ├── predictor.pth
    │   ├── predictor12.onnx
    │   └── trainingclassifier.py
    ├── preprocessing.py
    ├── run.py
    ├── trainer.py
    └── wandb_help.py

We note that data is not in this github repository and is in the link above.

/src usage:
After downloading data and adding to your enviornemnt

/models usage:
/models have some available models for immediate deployment, these are fine-tuned ViT-Tiny models described in this Repo's report.

We save a model for k1 usage (e.g. take a picture on campus and we can classify which building it is) labeled as predictor.pth. We transfer to .onnx for easy Unity and webpage usage. The script generalclassifer.py is capable of classifying any image as one of the 41 UCF buildings. You can classify by running:

python generalclassifer.py "/path/to/your/image.jpg"

usage of generalclassifer.py does not require the dataset

The script trainingclassifier.py is very similar to generalclassifier.py, where trainingclassifier.py randomly predicts a label from the test dataset and confirms whether it labeled correctly or not.

The file onnx_transfer.py gives the configuration of how we transfered our .pth file to a .onnx file.


