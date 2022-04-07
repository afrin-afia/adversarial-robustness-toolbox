import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
matplotlib.use('agg')

def data_diff():
    directory = os.getcwd()
    testData= pd.read_csv(directory+"/sgrid/outs/original_test_data.csv")
    fgsm_advData= pd.read_csv(directory+"/sgrid/outs/FGSM_adv_test_data.csv")
    pgd_advData= pd.read_csv(directory+"/sgrid/outs/PGD_adv_test_data.csv")

    #following: M*13 sized arrays
    #testDataArray= testData.values 
    #advDataArray= advData.values

    fgsm_diff= testData.subtract(fgsm_advData,axis=1)
    pgd_diff= testData.subtract(pgd_advData,axis=1)
    #print(diff.head)
    fgsm_diffArray= fgsm_diff.values
    pgd_diffArray= pgd_diff.values

    #MSE of differences i.e columnwise l2 norm----------FGSM
    s= fgsm_diffArray[0].size
    rows= np.shape(fgsm_diffArray)[0]
    avgd= np.empty(s-1)
    #print (avgd.size)
    for i in range(1,s):
        b= fgsm_diffArray[:,i]    #ith column of a
        #print(b)
        sum=0
        for j in range (b.size):
            bj= b.item(j)
            sum+= bj* bj
        avgd[i-1]= sum/rows
    #print (avgd)

    #plot barchart of MSE of differences (stored in avgd)
    barX= np.array(['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4', 'stab'])
    plt.bar(barX, avgd, align='center', alpha=0.5)
    plt.xlabel('Features')
    plt.ylabel('Average Perturbation (FGSM)')
    #plt.show()
    plt.savefig(directory+"/sgrid/outs/figs/FGSM_avg_perturbation_txt_data.png")

    
    #MSE of differences i.e columnwise l2 norm----- PGD
    s= pgd_diffArray[0].size
    rows= np.shape(pgd_diffArray)[0]
    avgd= np.empty(s-1)
    #print (avgd.size)
    for i in range(1,s):
        b= pgd_diffArray[:,i]    #ith column of a
        #print(b)
        sum=0
        for j in range (b.size):
            bj= b.item(j)
            sum+= bj* bj
        avgd[i-1]= sum/rows
    #print (avgd)

    #plot barchart of MSE of differences (stored in avgd)
    barX= np.array(['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4', 'stab'])
    plt.bar(barX, avgd, align='center', alpha=0.5)
    plt.xlabel('Features')
    plt.ylabel('Average Perturbation (PGD)')
    #plt.show()
    plt.savefig(directory+"/sgrid/outs/figs/PGD_avg_perturbation_txt_data.png")


data_diff()


    