# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%

# set parameters 

sigma= 0.25
lamda= 0.5
T= 40.0

NofPath=500
NofSet= 500

Z= np.random.normal(loc=0,scale=1,size=[NofPath,NofSet])
dt=T/NofSet

t=np.zeros(NofSet)
 

W= np.zeros([NofPath,NofSet])
R=  np.zeros([NofPath,NofSet])
P=  np.zeros([NofPath,NofSet])

P0=lambda x:np.exp(-lamda*x)
f0= lambda x : -(np.log(P0(x+dt))- np.log(P0(x-dt)))/(2*dt)

#set theta for HJM model
Theta = lambda x: (1/lamda)*((f0(x +dt) -f0(x-dt))/(2*dt))+ f0(x) + (sigma**2)*(1-np.exp(-2*lamda*x))/(2*lamda**2)


#Set affine parameter A and B at t=0 for the charcteristic function 

A_0=  lambda x: ((sigma**2)/(np.power(lamda,3)*4))*(3+ np.exp(-2*lamda*x)+ 4*np.exp(-lamda*x)-2*lamda*x) 
B_0= lambda x: -(1-np.exp(-lamda*x)) /lamda

# Add to A the deterministic path 

mu = lambda x: (1-np.exp(-lamda*x))


ro=f0(dt)
R[:,0]=ro
P[:,0]=1



# %%
def setParmaters():
    for i in range(NofSet-1):
        Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1]= W[:,i] + np.sqrt(dt)*Z[:,i]
        teta=Theta(t[i+1])
      
        R[:,i+1] = R[:,i] + lamda*( np.ones(NofPath)*teta -R[:,i] )* dt +sigma*(np.sqrt(dt)*Z[:,i])
        P[:,i+1]= P[:,i]*np.exp(-(R[:,i+1]+R[:,i])*0.5*dt)
        t[i+1]= t[i] +dt

    print("END")

# %%
setParmaters()
# %%

for i in range(NofPath):
    plt.plot(t,R[i,:])


#%%
for i in range(NofPath):
    plt.plot(t,P[i,:])

# %%

P_HJM=np.zeros(NofSet)
P_Mkt=np.zeros(NofSet)

for i in range(NofSet):
    P_HJM[i]=np.mean(P[:,i])
    P_Mkt[i]=P0(t[i])


plt.plot(t,P_HJM)
print(P_HJM)
plt.plot(t,P_Mkt)


# %%
