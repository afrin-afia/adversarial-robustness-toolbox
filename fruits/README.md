# Instructions for manual replication of this experiment

In this experiment, we observe the effects of adversarial attacks on two different image dataset. Required source codes are available inside the `src` folder. 

Replicating the results for the [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) dataset does not require any further modification. 
Simply run the `src/fruits.py` file to generate attacks against a sequential neural network from the `tensorflow.keras` library. Similarly, 
executing the `src/attacks_on_pytorch` file will generate attacks against a PyTorch CNN image classifier.

However, replicating the experiments on the 
[MNIST](https://ieeexplore.ieee.org/abstract/document/6296535?casa_token=fGZ6RD4tMY0AAAAA:0JP1BDQ-5Ga4YMc2Vnlg7e5hhUC1iTMPWJW6E3EGzFDYBgYH1xfICUwDcEwUhd0JvdVHZJ3y)
dataset requires an additional task to be done. Since the `MNIST` dataset is exceptionally large in volume and there already exists a github repository
for this dataset, we chose not to upload it again. Our source code uses the image version of the MNIST dataset, which can be found [here] 
(https://github.com/teavanist/MNIST-JPG). 

So, all you need to do is clone the dataset from the repository mentioned above and place it inside a folder named `fruits/mnistjpg`. All the training samples should be kept in a folder named `Training` and the test samples should be in a folder named `Test`. 

In a nutshell, the final directory tree should look like this:

    .
    ├── ...
    ├── fruits                    
    │   ├── mnistjpg         
    │     ├── Test         
    │     └── Training                
    └── ...

And... Voilà! You are all ready to rerun this experiment on MNIST dataset! 

Now, just execute the `src/mnist` python script. 
