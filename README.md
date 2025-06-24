# Plant Disease Classification

This repository contains a script to train a ResNet32 model on plant disease datasets focusing on potato, pepper and tulsi leaves.

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- scikit-learn

## Dataset Preparation

1. Download the following datasets manually (internet access is disabled in this environment):
   - Plant disease dataset focusing on potato and pepper: [link](https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded)
   - Tulsi leaf dataset: [link](https://www.kaggle.com/datasets/manjotkaurhpk/tulsi-leaf-train-and-test-dataset/data)
2. Extract the datasets into `data/original`.
3. Run the data preparation step to split into `train`, `valid`, and `test` directories and balance classes with augmentation:

```bash
python train_resnet32.py --data-dir data --prepare-data
```

This step should create the following structure:

```
data/
  train/
  valid/
  test/
```

Each class in the training set should contain 5000 images after augmentation (25% train, 25% valid, 50% test).

## Training

After data preparation, run the training for 100 epochs:

```bash
python train_resnet32.py --data-dir data --epochs 100
```

The script reports loss and accuracy for the validation set each epoch. After training, it evaluates the model on the test set and prints the confusion matrix, recall, precision and accuracy.

**Note:** The data preparation function is left unimplemented because it requires manual dataset processing and augmentation. Fill in the `prepare_data` function with your custom logic before running the script.

