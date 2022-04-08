# Instructions for manual replication of this experiment

In this experiment, we observe the effects of adversarial attacks on an audio dataset. Required source codes are available inside the `src` folder. 

Replicating the experiments on the 
[UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) dataset requires an additional task to be done. 
Since this dataset is a exceptionally large in volume and there already exists a publicly accessible URL to download the dataset, 
we chose not to upload it again. 

So, all you need to do is download the dataset from `[Kaggle]((https://www.kaggle.com/datasets/chrisfilo/urbansound8k))` 
and place it inside a folder named `urbansound/data`. 

The final directory tree should lokk like this:

    .
    ├── ...
    ├── urbansound                    
    │   ├── data         
    │     ├── fold1         
    │     └── fold2 
    |     ...
    |
    |     └── fold10   
    |     └── three_class.csv
    └── ...

And... Voilà! You are all ready to rerun this experiment on MNIST dataset! 

Now, 
* execute the `src/deepfool_audio_data_multiclass` python script to generate adversarial attacks.
* execute the `src/model_hardening_audio_data` python script to implement the model hardening defense.
* execute the `src/test_data_visualization` python script to enerate graphs.
