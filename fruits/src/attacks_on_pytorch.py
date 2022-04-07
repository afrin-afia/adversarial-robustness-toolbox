import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import cv2
import numpy as np
from art.estimators.classification import PyTorchClassifier
from keras.preprocessing import image
from art.attacks.evasion import FastGradientMethod, DeepFool

import os
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.optim import Adam

# load and preprocess imagenet images -- values in [0,1]
# images = load_preprocess_images(im_paths)
classes= ['Apple_Braeburn', 'Apricot', 'Avocado', 'Banana', 'Cherry', 'Guava', 'Lemon']

directory= os.getcwd() 

def createTrainData():
    
    images_list = list()
    y=list()
    c=0
    for classname in classes:

        path= directory+"/fruits/mydata/Training/"+classname+"/"
        img_files= os.listdir(path)         #an array of the names of all files
        for i in img_files:
            finalpath= path+i 
            im = image.load_img(finalpath, target_size=(100,100,3))
            im = image.img_to_array(im)
            #org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array((im/255).astype(np.float32).T))
            y.append(c)
        c= c+1 
    return images_list, y

def createTestData():

    images_list = list()
    y=list()
    c=0
    for classname in classes:
        path= directory+"/fruits/mydata/Test/"+classname+"/"
        img_files= os.listdir(path)         #an array of the names of all files
        for i in img_files:
            finalpath= path+i 
            im = image.load_img(finalpath, target_size=(100,100,3))
            im = image.img_to_array(im)
            #org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array((im/255).astype(np.float32).T))
            y.append(c)
        c= c+1 
    return images_list, y

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(2500, 7)#Linear(4 * 7 * 7, 7)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers (x)
        return x


model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()

images_list, y= createTrainData()
x_train_array= np.array(images_list)
y_train_array= np.array(y)
#shuffle data
p = np.random.permutation(len(x_train_array))
x_train_array= x_train_array[p]
y_train_array= y_train_array[p]
y_train= to_categorical(y_train_array)


images_list, y= createTestData()
x_test_array= np.array(images_list)
y_test_array= np.array(y)
p = np.random.permutation(len(x_test_array))
x_test_array= x_test_array[p]
y_test_array= y_test_array[p]
y_test= to_categorical(y_test_array)

num_classes= 7

#model= createModel() 
classifier = PyTorchClassifier(
    model=model,
    #clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 100, 100),
    nb_classes=7,
) #KerasClassifier(model=model)
classifier.fit(x_train_array, y_train, nb_epochs=12)
print("TRAINING DONE!")


#Test on benign data
preds = classifier.predict(x_test_array)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
#acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Model Accuracy on Benign Data: %.2f%%", (acc * 100))


#attack FGSM 
#create adversarial data
attack_fgsm = FastGradientMethod(classifier, eps=0.01) 
x_test_adv = attack_fgsm.generate(x_test_array)

#Test on adv data
preds =classifier.predict(x_test_adv)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Adversarial data(FGSM, eps= .01): %.2f%%", (acc * 100))


path= directory+"/fruits/inputOutput/apple_50_100.jpg"
im = cv2.imread(path)

org_im_in = [(im.T/255).astype(np.float32)]
#org_im_in= np.array(org_im_in)

#org_im_in= [org_im_in.astype(np.float32)]
org_im_in= np.array(org_im_in)
#org_im_in= org_im_in.T 
adv_images_df = attack_fgsm.generate(x = org_im_in)

prediction= classifier.predict(adv_images_df)
print("Original label: Apple, Predicted label (FGSM, eps= .01): "+ classes[np.argmax(prediction)])
cv2.imwrite(directory+"/fruits/inputOutput/out_fgsm_01_fruits_pytorch.jpg", 255*(adv_images_df[0].T))

###########

attack_fgsm = FastGradientMethod(classifier, eps=0.1) 
x_test_adv = attack_fgsm.generate(x_test_array)

#Test on adv data
preds =classifier.predict(x_test_adv)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Adversarial data(FGSM, eps= .1): %.2f%%", (acc * 100))


path=directory+"/fruits/inputOutput/apple_50_100.jpg"
im = cv2.imread(path)

org_im_in = [(im.T/255).astype(np.float32)]
#org_im_in= np.array(org_im_in)

#org_im_in= [org_im_in.astype(np.float32)]
org_im_in= np.array(org_im_in)
#org_im_in= org_im_in.T 
adv_images_df = attack_fgsm.generate(x = org_im_in)

prediction= classifier.predict(adv_images_df)
print("Original label: Apple, Predicted label (FGSM, eps= .1): "+ classes[np.argmax(prediction)])
cv2.imwrite(directory+"/fruits/inputOutput/out_fgsm_1_fruits_pytorch.jpg", 255*(adv_images_df[0].T))


attack = DeepFool(
    classifier=classifier,
    
)
x_test_adv = attack.generate(x_test_array)
preds =classifier.predict(x_test_adv)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Adversarial data(DeepFool): %.2f%%", (acc * 100))

path=directory+"/fruits/inputOutput/apple_50_100.jpg"
im = cv2.imread(path)

org_im_in = [(im.T/255).astype(np.float32)]
#org_im_in= np.array(org_im_in)

#org_im_in= [org_im_in.astype(np.float32)]
org_im_in= np.array(org_im_in)
#org_im_in= org_im_in.T 
adv_images_df = attack.generate(x = org_im_in)

prediction= classifier.predict(adv_images_df)
print("Original label: Apple, Predicted label (DeepFool): "+ classes[np.argmax(prediction)])
cv2.imwrite(directory+"/fruits/inputOutput/out_deepFool_fruits_pytorch.jpg", 255*(adv_images_df[0].T))








