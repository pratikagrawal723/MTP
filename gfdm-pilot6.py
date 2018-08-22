
# coding: utf-8

# In[1]:


# (m,k) convention : interferers are neighbours
# values in paper

import numpy as np
from numpy import * 
from numpy import pi
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy import signal
import filters
from numpy.linalg import inv
from scipy.stats import threshold

#-------------------------------------------------------------------------------------------------------
M=5          # no of symbols/ sub-carrier
K=128          # no of subcarriers
del_k=4
N=M*K
SNR_dB=20

pilotValue = 3-3j                        # The known value each pilot transmits
allIndice = np.arange(N)   
pilotI= allIndice[::(N//(del_k*M))]
pilotI= allIndice[::del_k*M]
pilotIndice=[]
for i in range(0,len(pilotI)):
    for m in range(0,M):
        pilotIndice.append(pilotI[i]+m) 
    
dataIndice= np.delete(allIndice, pilotIndice)

#print ("allIndice: ",allIndice)
#print ("pilotIndice: ",pilotIndice)
#print ("dataIndice: ",dataIndice)
lp = len(pilotIndice)
#----------------------------------------------------------------------------------------------------------

mu = 4                                                            # bits per symbol (i.e. 16QAM)  Mapping
payloadBits_per_GFDM = len(dataIndice)*mu                         # number of payload bits 

bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_GFDM, ))  # Transmission Bits


# In[2]:


## Mapping : Gray labelling
mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
demapping_table = {v : k for k, v in mapping_table.items()}    # inverse of mapping table

def SP(bits):           # Serial to parallel convertor
    return bits.reshape((len(dataIndice), mu))
def Mapping(bits):      # Mapper mu
    return np.array([mapping_table[tuple(b)] for b in bits])

bits_SP = SP(bits)
QAM = Mapping(bits_SP)


# In[3]:


def GFDM_symbol(QAM_payload):
    symbol = np.zeros(N, dtype=complex) # the overall K subcarriers
    symbol[pilotIndice] = pilotValue    # allocate the pilot subcarriers 
    symbol[dataIndice] = QAM_payload    # allocate the data subcarriers
    return symbol

GFDM_data = GFDM_symbol(QAM)   # OFDM data in freq domain
print ("Number of GFDM data in frequency domain: ", len(GFDM_data))
#print ("GFDM Data: ",GFDM_data)
d=GFDM_data       # d = GFDM Data Symbols


# In[4]:


def GFDM_modulation_matrix(g):     # GFDM Modulation Matrix
    Al=[]
    G_indices=[]
    for n in range(0,N):
        for k in range(0,K):
            for m in range(0,M):
                indice= (n-m*K)%N
                term=(2*pi*k*n)/K
                val=g[indice]*np.exp(1j*term)
                #print("k,m",k,m)
                Al.append(val)
                if(n==0):
                    ind=(m,k)
                    G_indices.append(ind)
                          
    At=np.asarray(Al)
    G=reshape(At,(N,N))
    return (G,G_indices)

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * pi * 1J / N )
    W = np.power( omega, i * j ) / sqrt(N)
    return W

def noise_vector(signal_power):
    n_var = signal_power * 10**(-SNR_dB/10)  # calculate noise power
    w1=np.random.normal(0,1,N)               # complex gaussian noise
    w2=np.random.normal(0,1,N)
    w=((np.sqrt(n_var / 2))*(w1+1j*w2))
    return w


# In[5]:


alpha=0.5                               # Raised cosine (RRC) filter
Fs=40000000;Ts=0.000002
#Fs=40000;Ts=0.00005
time_idx, g =filters.rrcosfilter(N, alpha, Ts, Fs)
g=np.roll(g, N//2)
#print("Filter Coefficients",g)

G,G_indices = GFDM_modulation_matrix(g)      # N x N
#G = threshold(G, e-16)
#print("G Indices", G_indices)

#print(np.linalg.det(G))
plt.plot(g)
plt.grid(True)
plt.show()


# In[6]:


def interference_at_pliot(p_index):
    p_value = d[p_index]
    pI_index = find_interferer(p_index)
    #print(pI_index)
    interference=0        # p_index, pI_index, pI
    
    for pI in pI_index:
        ambiguity_p = ambiguity(p_index,pI)             # send the pilot index and interfereing pilot index
        inter = d[pI]*ambiguity_p                      # equation 13
        print("Interference from ",pI,"to",p_index,"is",inter)
        interference = interference + inter
    return interference


def find_interferer(p_index):                  # find the interferers
    pI_index=[]
    for i in pilotIndice:
        if( abs(i-p_index) <= (M-1)):
            pI_index.append(i)
    pI_index.remove(p_index)
    
    return pI_index  


def ambiguity(p_index,pI):                              # ambiguity function (interferer pilot and original pilot*)
    ambiguity=0
    for n in range(0,N):
        amb = G[n][pI] * np.conjugate( G[n][p_index] )
        ambiguity = ambiguity + amb
    return ambiguity
    


# In[7]:


#Interference at pilots
Interference = []
for s in pilotIndice: 
    val = interference_at_pliot(s) 
    Interference.append(val)

print("Original pilots:",d[pilotIndice])
d_original = d[pilotIndice]

for s in range(0,lp): 
    d[pilotIndice[s]] = d[pilotIndice[s]] - Interference[s]

print("Precoded pilots:",d[pilotIndice])
#print("\n")
#print("All data", d)


# In[8]:


x=np.matmul(G,d)        # GFDM Modulate

#-------------- Channel ---------------------------------

h = np.array([1,0.3+0.3j])          # Channel coefficients
#h= np.array([1])
#h = np.array([1,0.9j,0.7])          # Channel coefficients
Nch=len(h)          
zeros= np.zeros(N-Nch)
hp=np.hstack([h,zeros])            
H=circulant(hp)

y=np.matmul(H,x)   

signal_power = np.mean(abs(y**2))
w = noise_vector(signal_power)

yt = y+w                                 # Add noise   # Final received signal 
#print("y",y)
#print("yt",yt)
#print(w)


# In[9]:


# Channel in frequency, over the symbols ----------------
H_exact = np.fft.fft(h, N)
plt.plot(allIndice, abs(H_exact))
plt.grid(True)
plt.show()


# In[10]:


#-------- Receiver --------------------------------------

#--Matched filter

Gm = np.matrix(G)
G_conj = Gm.H
G_conj = np.array(G_conj)

# filtering y with g*[m,n] associated with pilot symbols
z_mk = []

for p in pilotIndice:
    z=0
    for i in range(0,N):
        rx = yt[i] * G_conj[p][i]
        z = z + rx
    z_mk.append(z)
    
#z_mk


# In[11]:


#------- Estimation ---------------------------------
#estimate channel coefficient at PILOT Position

H_est = z_mk/d_original
#H_est = z_mk/d[pilotIndice]

#print("Estimated H:",H_est)
H_exact = np.fft.fft(h, lp)
#print("Original H:",H_exact)
#print("Scaling:",H_est/H_exact)


# In[12]:


Gm= np.absolute(np.real(G))
a=np.amin(Gm)
print("Minimum value in GFDM Matrix:",a)
print("Rank with threshold:",np.linalg.matrix_rank(G,tol=e-5))
print("Condition no.:",np.linalg.cond(G))


# In[24]:


H_exact = np.fft.fft(h, N)
plt.plot(allIndice, abs(H_exact), label='Correct Channel')

plt.plot(pilotIndice, abs(H_est)/1100, label='Pilot estimates')

plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

