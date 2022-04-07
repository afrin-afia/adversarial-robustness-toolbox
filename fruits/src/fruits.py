import tensorflow as tf
import keras
tf.compat.v1.disable_eager_execution()
import cv2
import numpy as np
from keras.preprocessing import image

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

import os
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# load and preprocess imagenet images -- values in [0,1]
# images = load_preprocess_images(im_paths)
classes= ['Apple_Braeburn', 'Apricot', 'Avocado', 'Banana', 'Cherry', 'Guava', 'Lemon']

directory= os.getcwd()

def createTrainData():
    
    images_list = list()
    y=list()
    c=0
    for classname in classes:

        path=directory+"/fruits/mydata/Training/"+classname+"/"
        img_files= os.listdir(path)         #an array of the names of all files
        for i in img_files:
            finalpath= path+i 
            im = image.load_img(finalpath, target_size=(100,100,3))
            im = image.img_to_array(im)
            org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array(org_im_in))
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
            org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array(org_im_in))
            y.append(c)
        c= c+1 
    return images_list, y

def createModel():
    #classification model 
    model = Sequential() 
    model.add(Flatten(input_shape=x_train_array.shape[1:])) 
    model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.5)) 
    model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.3)) 
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc'])
    return model 


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

model= createModel() 
classifier = KerasClassifier(model=model)
classifier.fit(x_train_array, y_train, nb_epochs=8)
print("TRAINING DONE!")


#Test on benign data
preds = classifier.predict(x_test_array)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
#acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Model Accuracy on Benign Data: %.2f%%", (acc * 100))


#attack FGSM 
#create adversarial data
attack_fgsm = FastGradientMethod(classifier, eps=0.2) #ProjectedGradientDescent(classifier, targeted=False, max_iter=10, eps_step=1, eps=5)#
x_test_adv = attack_fgsm.generate(x_test_array)

#Test on adv data
preds =classifier.predict(x_test_adv)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Adversarial data(FGSM): %.2f%%", (acc * 100))



#see random single image
# Generate the adversarial sample:
path= directory+"/fruits/inputOutput/apple_50_100.jpg"
im = cv2.imread(path)

org_im_in = [(im/255).astype(np.float32)]
org_im_in= np.array(org_im_in)

org_im_in= [org_im_in.astype(np.float32)]
adv_images_df = attack_fgsm.generate(x = np.array(org_im_in))

prediction= classifier.predict(adv_images_df)
print("Original label: Apple, Predicted label (FGSM): "+ classes[np.argmax(prediction)])
cv2.imwrite(directory+"/fruits/inputOutput/out_fgsm_fruits.jpg", 255*adv_images_df[0][0])


################
##attack PGD
#Test on adv data
attack_pgd = ProjectedGradientDescent(classifier, targeted=False, max_iter=10, eps_step=1, eps=5)
x_test_adv = attack_pgd.generate(x_test_array)
preds =classifier.predict(x_test_adv)
acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Adversarial data(PGD): %.2f%%", (acc * 100))



#see random single image
# Generate the adversarial sample:
path= directory+"/fruits/inputOutput/apple_50_100.jpg"
im = cv2.imread(path)

org_im_in = [(im/255).astype(np.float32)]
org_im_in= np.array(org_im_in)

org_im_in= [org_im_in.astype(np.float32)]
adv_images_df = attack_fgsm.generate(x = np.array(org_im_in))

prediction= classifier.predict(adv_images_df)
print("Original label: Apple, Predicted label(PGD): "+ classes[np.argmax(prediction)])
cv2.imwrite(directory+"/fruits/inputOutput/out_pgd_fruits.jpg", 255*adv_images_df[0][0])

print(" ")
print("DONE! Check 'inputOutput' folder for sample outputs :) ")










