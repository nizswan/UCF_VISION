# UCF VISION
Publically Available Dataset(s): https://drive.google.com/drive/folders/1IUVSkOcDjggGhobZBP3_V69BegmhDnkp?usp=drive_link 

The true tree from the directory of training should look like:
```
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
```

We note that data is not in this github repository and is in the link above.

/src usage:
After downloading data and adding to your enviornemnt we can train.

wandb_help.py is a generic helper script that allows for easy logging to wandb for easy metric collection.

preprocessing.py consists of how we preprocessed our data (scaling to 224x224 and using various distortion techniques)

distance_matrix.py stores a single object based on calculations computed outside of this scope. It is the matrix which identifies the distances between ith and jth class.

trainer.py is the training script used for training and storing models

run.py acts as a convenient parser of inputs

You can train after saving the data folder by the following command.

```
(ViT-Tiny, basic preprocessing, k=1 dataset, lambda = 0.1, learning rate = 3*10^-4, batch size = 64, epochs = 30)
sbatch run.slurm -m tiny -p base -k 1 -L 0.1 -r 3e-4 -b 64 -e 30
```

/models usage:
/models have some available models for immediate deployment, these are fine-tuned ViT-Tiny models described in this Repo's report.

We save a model for k1 usage (e.g. take a picture on campus and we can classify which building it is) labeled as predictor.pth. We transfer to .onnx for easy Unity and webpage usage. The script generalclassifer.py is capable of classifying any image as one of the 41 UCF buildings. You can classify by running:

```
python generalclassifer.py "/path/to/your/image.jpg"
```

usage of generalclassifer.py does not require the dataset

The script trainingclassifier.py is very similar to generalclassifier.py, where trainingclassifier.py randomly predicts a label from the test dataset and confirms whether it labeled correctly or not.

The file onnx_transfer.py gives the configuration of how we transfered our .pth file to a .onnx file.



