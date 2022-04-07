
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import DeepFool

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import Dense, Activation, Dropout
import os 
import librosa
import librosa.display
import pandas as pd
from tqdm import tqdm


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features, sample_rate


def createDfFromData():
    extracted_features=[]
    csvFile= directory+"/urbansound/data/three_class.csv"          #.csv file is shuffled
    
    metadata=pd.read_csv(csvFile)
    filePath= directory+"/urbansound/data"
    i=0
    for index_num,row in tqdm(metadata.iterrows()):
        file_name = filePath+ '/fold'+ str(row["fold"])+ '/'+ str(row["slice_file_name"])
        final_class_labels= row["class"]
        data, sample_rate=features_extractor(file_name)
        extracted_features.append([data,final_class_labels])
        i= i+1
       
    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    
    return extracted_features_df, sample_rate


directory= os.getcwd()

extracted_features_df, sampling_rate= createDfFromData()
X= np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

#class names are strings (gun_shot, dog_bark). Make them numerical
labelencoder=LabelEncoder()
#Transform the numbers into vectors [0 0 1] or [1 0 0]
y=to_categorical(labelencoder.fit_transform(y))

#split train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


#num_epochs = 200
#num_batch_size = 32

# Create classifier wrapper
classifier = KerasClassifier(model=model)
classifier.fit(X_train, y_train, nb_epochs=100, batch_size=32)
print("TRAINING DONE!")

#Test on benign data
preds = np.argmax(classifier.predict(X_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
#logger.info("Classifier before adversarial training")
print("Model Accuracy: %.2f%%", (acc * 100))

#create adversarial data
adv_crafter = DeepFool(classifier)
x_test_adv = adv_crafter.generate(X_test)

#Test on adv data
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
#logger.info("Classifier before adversarial training")
print("Model Accuracy on adversarial data: %.2f%%", (acc * 100))

### WRITE SAMPLING RATE IN A FILE SO WE CAN USE IT FROM highest_purturbed_data.py ###

f = open(directory+"/urbansound/outs/samplingRate.txt", "a")
f.truncate(0)
f.write(str(sampling_rate))
f.close()


