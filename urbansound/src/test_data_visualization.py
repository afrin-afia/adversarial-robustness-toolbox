import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import pandas as pd

import librosa
import librosa.display
import os 
import soundfile as sf

directory= os.getcwd()
print(directory)
testData= pd.read_csv(directory+"/urbansound/outs/original_test_data.csv")
advData= pd.read_csv(directory+"/urbansound/outs/adv_test_data.csv")

def data_diff():    
    #following: M*40 sized arrays
    #testDataArray= testData.values 
    #advDataArray= advData.values

    diff= testData.subtract(advData,axis=1)
    
    diffArray= diff.values

    #MSE of differences i.e columnwise l2 norm
    s= diffArray[0].size
    rows= np.shape(diffArray)[0]
    avgd= np.empty(s-1)
    #print (avgd.size)
    for i in range(1,s):
        b= diffArray[:,i]    #ith column of a
        sum=0
        for j in range (b.size):
            bj= b.item(j)
            sum+= bj* bj
        avgd[i-1]= sum/rows
    #print (avgd)

    #plot barchart of MSE of differences (stored in avgd)
    
    #barX= np.array(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 
    #'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 
    #'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40'])
    
    #plt.bar(barX, avgd, align='center', alpha=0.5)
    #plt.show()
    return avgd
    


avg_dist= data_diff()           #array. avg dist of each features 

#add this amount to original data. then show the waveform and generate audio file!
file_name= directory+"/urbansound/data/fold1/9031-3-3-0.wav"

audio, sampling_rate = librosa.load(file_name)

fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.waveshow(audio,sr=sampling_rate, ax=ax[0])
ax[0].set(title='Waveform: original data sample')
ax[0].label_outer()

#ipd.Audio(file_name)



mfccs_features = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=40).T
#print(mfccs_features.shape)  #173 * 40

for i in range (0, mfccs_features.shape[0]):            #o to 172
    mfccs_features[i]= mfccs_features[i]+ avg_dist

adv_audio= librosa.feature.inverse.mfcc_to_audio(mfccs_features.T, n_mels=50)
sf.write(file=directory+"/urbansound/outs/adversarial_data.wav", data=adv_audio, samplerate=sampling_rate)

librosa.display.waveshow(adv_audio,sr=sampling_rate, ax=ax[1])
ax[1].set(title='Waveform: adversarial data sample')
ax[1].label_outer()
#plt.show()
plt.savefig(directory+"/urbansound/outs/figs/waveforms_audio_data.png")
