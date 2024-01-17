# Barlow Twins for Lung Neuroendocrine Neoplasms (LNENs): 
Unsupervised deep learning model trained to extract features from images. The adaptation we propose here is dedicated to learning the features of the tiles making up whole slide images of Lung Neuroendocrine Neoplasms (LNENs). The encoded vectors created by the Barlow twins of tiles sharing common pathological features are assumed to be closer in latent space than less similar tiles.

- Original article: J. Zbontar [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html), PMLR 2021.
- Original code: [https://github.com/facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins)
- Method used to extract features from WSIs stained with haematoxylin and eosin (HE) to distinguish typical from atypical carcinoids in "Assessment of the current and emerging criteria for the histopathological classification of lung neuroendocrine tumours in the lungNENomics project." ESMO Open 2023 (under review)

## Installation
- Clone this repository: tested on Python 3.9
- Install [PyTorch](http://pytorch.org/): tested on v1.9.0
- Install [Torchvison](https://pytorch.org/vision/stable/index.html) tested on v0.10.0
- Install [cudatoolkit](https://developer.nvidia.com/cuda-toolkit) tested on 10.2.0
- Install [pillow](https://pillow.readthedocs.io/en/stable/)  tested on 9.3.0
- Install any version of numpy
- Other dependencies in environment.yml

Install all packages with this command:
```
$ conda env create -f environment.yml
```

## Dataset
This model was trained on 259 HE-stained  WSIs of LNEN. The WSIs were cut into 384x384 pixel tiles and the colors were normalized using Vahadane's color deconvolution method. Pre-processing scripts are available in  [https://github.com/IARCbioinfo/WSIPreprocessing](https://github.com/IARCbioinfo/WSIPreprocessing). The ~4.1M pre-processed tiles aer available on request from mathiane@iarc.who.int.

## Training Model
- An example of the configurations used to trained Barlow Twins for LNEN WSIs is available in `Bash/Train/TumorNormal/TrainToyDataset.sh`
- The commands below are used to train the model based on the toy data set:
```
bash Bash/Train/TrainToyDataset.sh 
```
- **Note:** 
  **- Additional configurations and parameters are described at the beginning of`main.py`.**
  **- Trained network weights are available on request.**

## Testing Pretrained Models
- Download pretrained weights are available on request and will be soon available online 
- An example of the configurations used to infer the test set is gien in `Run/Test/TumorNormal/TestToyDataset.sh`
```
bash Run/Test/TumorNormal/TestToyDataset.sh
```
- Main configurations:
    + checkpoint: Path to model weights to be loaded to infer the test tiles.
    + viz-dir: Directory where the result table will be saved.
    + viz-anom-map: If specified, all anomaly maps will be written to the `viz-dir` directory in `.npy` format.

## Results exploration
For each tile, `results_table.csv` summarises:
- Its path, which may include the patient ID
- Binary tile labels, useful for sorted datasets: Tumour = 2 and Non-tumour = 1 
- Max anomaly scores: value of the highest anomaly score of the tile
- Mean anomaly scores: average anomaly score of the tile

**The distributions of these score are used to segment the WSI.**

An example of result exploration for the segmentation of HE/HES WSI is given in `ExploreResultsHETumorSeg.html`.

## Get tumor segmentation map 

The `TumorSegmentationMaps.py` script is used to create the tumour segmentation map for a WSI. An example configuration is given in `ExRunTumorSegmentationMap.sh`. The results of this script are stored in the `Example_SegmentationMap_PHH3` folder, which also gives an example of the model's performance in segmenting a PHH3-immunostained WSI.

## TO DO LIST

+ :construction: Check parallel training 
+ :construction: Check parallel test
+ :construction: Model checkpoints Ki-67 and HES/HE