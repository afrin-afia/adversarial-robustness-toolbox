# Adversarial Robustness of Contemporary Machine Learning Models

This is my final project repository for the CMPUT 664 course at the University of Alberta in the winter of 2022. 

## Author
Afia Afrin

## Project Description

In this work, we have implemented three different adversarial attacks on four different datasets and the model hardening defense mechanism on two datasets. We leveraged the open-source tool, [`Adversarial Robustness Toolbox`](https://github.com/Trusted-AI/adversarial-robustness-toolbox), to generate these attacks. We utilized the following datasets-

* [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
* [MNIST](https://github.com/teavanist/MNIST-JPG)
* [Smart Grid Stability](https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability)
* [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

This repository includes:

* a [Dockerfile](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/Dockerfile) to build the Docker script,
* the instructions to reproduce the results
* `python` source codes to generate:
  * FGSM and PGD attacks on text dataset ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/sgrid/src/fgsm_pgd_text_data.py))
  * Defense mechanism: model hardening with PGD on text dataset ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/sgrid/src/model_hardening_text_data.py))
  * DeepFool attack on audio dataset ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/urbansound/src/deepfool_audio_data_multiclass.py))
  * Defense mechanism: model hardening with PGD on audio dataset ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/urbansound/src/model_hardening_audio_data.py))
  * FGSM and DeepFool attacks on image dataset (Fruits-360, Model: CNN, PyTorch) ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/fruits/src/attacks_on_pytorch.py))
  * FGSM and PGD on image dataset (Fruits-360, Model: sequential NN, tf.keras) ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/fruits/src/fruits.py))
  * FGSM and DeepFool on image dataset (MNIST, Model: sequential NN, tf.keras) ([here](https://github.com/afrin-afia/adversarial-robustness-toolbox/blob/main/fruits/src/mnist.py))
  * Output files, graphs, plots, and output images

## Docker Image
A pre-built version of this project is available as Docker image. To reproduce the experimental results using first clone this repository using the `git clone` command.
```
git clone <this repository>
```

This repository includes the `smart grid stability` text dataset and the `fruits-360` image dataset. However, you need to place the `MNIST` and `UrbanSound8K` datasets inside the appropriate directories. This datasets are so large that we skipped uploading them here. Just follow the directions provided [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/fruits) and [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/urbansound) to place the datasets in proper locations. 

Then build and run the docker image to reproduce the results.

```
docker build -t <image_name>
docker run -i <image_name>
```
A demo video showing how to reproduce the experiments using the pre-built Docker image is available in this [`folder`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/demo%20video-%20reproducing%20results). 

## Manual Installation
For manual installation process follow the guide provided in the [`ART git repository`](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started#setup). While reproducing the experiments using manual setup, kindly follow the instructions provided in the corresponding `README` files. 

* Instructions to manually replicate the experiments on image dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/fruits).

* Instructions to manually replicate the experiments on text dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/sgrid).

* Instructions to manually replicate the experiments on audio dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/urbansound).


## Experimental Results
Experimental results and outputs from the attacks on image dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/fruits/inputOutput).

Experimental results and outputs from the attacks on text dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/sgrid/outs).

Experimental results and outputs from the attacks on audio dataset are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/urbansound/outs).

## Documents
The final project report and presentation slide are available [`here`](https://github.com/afrin-afia/adversarial-robustness-toolbox/tree/main/Documents).

## Acknowledgement
I acknowledge that all external resources have been listed and referred to appropriately. I would also like to express my thankfulness to our course instructor Professor [Dr. Karim Ali](https://karimali.ca/). 
