import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
##This file finds which data has been perturbed the highest.
##We use l2 norm to measure distance/perturbation. 

def data_diff(testData, advData):
    

    #following: M*40 sized arrays
    #testDataArray= testData.values 
    #advDataArray= advData.values

    diff= testData.subtract(advData,axis=0)
    #print(diff.head)
    diffArray= diff.values

    #MSE of differences i.e columnwise l2 norm
    s= diffArray[0].size    #number of columns in diffArray
    rows= np.shape(diffArray)[0] #number of rows in diffArray

    avgd= np.empty(rows-1)
    #print (avgd.size)
    for i in range(1,rows):
        b= diffArray[i]    #ith row of diffarray
        #print(b)
        sum=0
        for j in range (b.size):
            bj= b.item(j)
            sum+= bj* bj
        avgd[i-1]= sum/s
    #print (avgd.shape)         #it's shape is 412, because we have 412 test data
    return avgd



directory = os.getcwd()
testData= pd.read_csv(directory+"/sgrid/outs/original_test_data.csv")
fgsm_advData= pd.read_csv(directory+"/sgrid/outs/FGSM_adv_test_data.csv")
pgd_advData= pd.read_csv(directory+"/sgrid/outs/PGD_adv_test_data.csv")
fgsm_avg_distance= data_diff(testData, fgsm_advData)   #an array of size 412 (number of test data)
pgd_avg_distance= data_diff(testData, pgd_advData)

fgsm_index_max= np.argmax(fgsm_avg_distance)  #which data has highest perturbation? Get it's index
pgd_index_max= np.argmax(pgd_avg_distance) 

#print(index_max)
#print(testData.head(10))
testDataArray= testData.to_numpy()
#print(testDataArray.shape)             413 *41
fgsm_advDataArray= fgsm_advData.to_numpy()
pgd_advDataArray= pgd_advData.to_numpy()

#print(advDataArray[index_max])
fgsm_x= testDataArray[fgsm_index_max].size
pgd_x= testDataArray[pgd_index_max].size

epochs= np.array(['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4', 'stab'])          # X-axis values
plt.scatter(epochs, testDataArray[fgsm_index_max][1:], color="blue", label="True Test Data")
plt.scatter(epochs, fgsm_advDataArray[fgsm_index_max][1:], color="red", label= "Adversarial Data (FGSM, eps= 0.2)")
plt.xlabel('Features')
plt.ylabel('Value')
for i in range (1,epochs.size+1):
    X= [ i-1,  i-1]
    Y= [ testDataArray[fgsm_index_max][i], fgsm_advDataArray[fgsm_index_max][i] ]
    plt.plot(X,Y)
plt.legend()
#plt.show()
plt.savefig(directory+"/sgrid/outs/figs/FGSM_most_perturbed_txt_data.png")

plt.scatter(epochs, testDataArray[pgd_index_max][1:], color="blue", label="True Test Data")
plt.scatter(epochs, pgd_advDataArray[pgd_index_max][1:], color="red", label= "Adversarial Data (PGD, eps= 0.2)")
plt.xlabel('Features')
plt.ylabel('Value')
for i in range (1,epochs.size+1):
    X= [ i-1,  i-1]
    Y= [ testDataArray[pgd_index_max][i], pgd_advDataArray[pgd_index_max][i] ]
    plt.plot(X,Y)
plt.legend()
#plt.show()
plt.savefig(directory+"/sgrid/outs/figs/PGD_most_perturbed_txt_data.png")
