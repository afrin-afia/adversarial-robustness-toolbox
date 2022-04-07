
#!/usr/bin/python


c= "Enter the number of experiment you want to run-\n"
c1= "1. FGSM & PGD attack on text data\n"
c2= "2. Defense: model hardening with PGD on text data\n"
c3= "3. Generate perturbation graph of the maximum perturbed test data sample (text)\n"
c4= "4. Average perturbation by FGSM attack on text data\n"

c5= "5. DeepFool attack on audio data\n"
c6= "6. Defense: model hardening with PGD on audio data\n"
c7= "7. Generate waveforms and adversarial audio sample\n"
c8= "8. FGSM & DeepFool on image data (fruits dataset, library: PyTorch)\n"
c9= "9. FGSM & PGD on image data (fruits dataset, library: Tensorflow.Keras)\n"
c10= "10. FGSM & DeepFool on image data (MNIST dataset, library: Tensorflow.Keras)\n"
b= "Or, press 'z' to end: "

c= c+c1+c2+c3+c4+c5+c6+c7+c8+c9+c10+b

while (1):

    choice= input(c)

    if (choice=="z"):
        print("\n\n\nHave A Good Day!\n")
        break

    elif (choice=="1"):
        import sgrid.src.fgsm_pgd_text_data

    elif (choice== "2"):
        import sgrid.src.model_hardening_text_data

    elif (choice== "3"):
        import sgrid.src.highest_perturbed_data

    elif (choice== "4"):
        import sgrid.src.test_data_visualization

    elif (choice== "5"):
        import urbansound.src.deepfool_audio_data_multiclass

    elif (choice== "6"):
        import urbansound.src.model_hardening_audio_data

    elif (choice== "7"):
        import urbansound.src.test_data_visualization

    elif (choice== "8"):
        import fruits.src.attacks_on_pytorch
    elif (choice== "9"):
        import fruits.src.fruits

    else:
        import fruits.src.mnist

  
