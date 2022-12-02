'''
probability calibration
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prob_cal(prediction_arr,target_arr):
    '''
    prob_cal(df,prediction_col,target_col)
    df - dataframe with the predictions 
    '''
    print(f"Probability calibration shapes must align {prediction_arr.shape,target_arr.shape}")

    fig , (ax1,ax2) = plt.subplots(ncols=2,figsize= (8,5))
    
    ax1.hist(prediction_arr, color = 'green')
    ax1.set_title('Histogram of Predicted Values')
    ax1.set_xlabel('prediction_arr')
    #ax1.set_ylabel('Estimated Salary')

    ax2.hist(target_arr, color = 'red')
    ax2.set_title('Histogram of Target Values')
    ax2.set_xlabel('target_arr')
    #ax2.set_ylabel('Estimated Salary')
    
    plt.show()
    
    comb = np.concatenate((prediction_arr,target_arr),axis=1)
    df = pd.DataFrame(comb)
    df.sort_values(0,inplace=True)
    df.reset_index(drop=True,inplace=True)
    df[2] = df[0].round(1)
    div = len(prediction_arr)/10
    df[3] = np.floor(np.arange(0,(df.shape[0])/div,1/div))[:prediction_arr.shape[0]] #ensures the same length as the other arrays
    print(df[3])
    
    tmp1 = df[[0,1,2]].groupby(2).mean()
    tmp2 = df[[0,1,3]].groupby(3).mean()
    
    fig , (ax1,ax2) = plt.subplots(ncols=2,figsize= (8,5))
    
    ax1.scatter(tmp1[0],tmp1[1],color = 'green')
    ax1.set_title('Predicted Values vs Probabilities by Rounding')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Target Values')

    ax2.scatter(tmp2[0],tmp2[1], color = 'red')
    ax2.set_title('Predicted Values vs Probabilities by Grouping')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Target Values')
    
    plt.show()
   
    
# do some testing
if __name__ == '__main__':
    #app.run_server()
    p = np.random.random((10000,1))
    targ = np.random.binomial(1,p)
    prob_cal(p,targ)
    
    






'''
np.arange(0,10,0.1)
np.arange(0,103/5,1/5)
np.arange(0,(103+1)/5,1/5)
np.floor(np.arange(0,(103+1)/5,1/5))
np.random
?np.random
?np.random.rand
np.random.rand((100,1))
?np.random.rand
np.random.rand(100,1)
np.random.rand(100,)
a = np.random.rand(100,1)
a1 = np.random.rand(100,)
a.shape
a1.shape
a.dim
a.dim()
dim(a)
a.shape
a1.shape
len(a1.shape)
len(a.shape)
a = np.random.rand(100,1)
b = np.random.randint(100,1)
b = np.random.randint(1,size=(100,1))
b
b = np.random.randint(2,size=(100,1))
b
b.mean()
b = np.random.randint(2,size=(100,1))
b.mean()
a = np.random.rand(103,1)
b = np.random.randint(2,size=(103,1))
a.shape
a.mean()
b.shape
b.mean()
import pandas as pd
df = pd.DataFrame(data=[a,b])
np.array(a,b)
np.array((a,b))
np.array([a,b])
a
np.concatenate((a,b),axis=1)
b = np.random.randint(2,size=(103,))
a = np.random.rand(103,)
a
b
np.concatenate((a,b),axis=1)
np.concatenate((a,b),axis=0)
np.concatenate((a,b),axis=2)
a = np.random.rand(103,1)
b = np.random.randint(2,size=(103,1))
comb = np.concatenate((a,b),axis=2)
comb = np.concatenate((a,b))
comb
comb.shape
comb = np.concatenate((a,b),axis=1)
comb.shape
comb
a[:,0]
a[:,1]
a[:,0]
comb[:,1]
comb[:,0]
comb[0,:]
df = pd.DataFrame(comb)
df
df[0].round
df[0].round(1)
df[0].round(2)
df[0].round(1)
df
df[2] = df[0].round(1)
df
df.groupby(2)
df.groupby(2).mean()
df
df.sort_values(0)
df.sort_values(0,inplace=True)
df
df.sort_values(0,inplace=True,reset_index=True)
df
df.reset_index()
df.reset_index(drop=True)
df.reset_index(drop=True,inplace=True)
df
df.shape
df.shape[0]
np.floor(np.arange(0,(df.shape[0]+1)/5,1/5))
np.floor(np.arange(0,(df.shape[0]+1)/5,1/5)).shape
np.floor(np.arange(0,(df.shape[0])/5,1/5)).shape
np.floor(np.arange(0,(df.shape[0])/5,1/5))
df
df[3] = np.floor(np.arange(0,(df.shape[0])/5,1/5))
df
df.groupby(2).mean()
df.groupby(3).mean()
df
'''