# Name: Sohag Kumar Saha
# Assignment-2 (Problem 3) 
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 

#import numpy, matplotlib library

import numpy as np
import matplotlib.pyplot as plt

# initialize values
N = 8
n = np.arange(N)
x = 0.1 * n

# Define Function for Discrete Fourier Transform  - DFT
def DFT_Function(x):
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * (1/(np.sqrt(N)))  * np.exp((1j *2* np.pi * k * n) / N) # Equation of DFT
    return X

# Define Function for Discrete Consine Transform - DCT
def DCT_Function(x):
    X = np.zeros(N)
    for k in range(N):  
        for n in range(N):          
            if k == 0:
                c = 1 / np.sqrt(N) # at k=0 
            else:
               # c = np.sqrt(2 / N) 
                c = np.sqrt(2 / N) * np.cos(((np.pi)/N) * (n + 0.5) * k )
            

            #X[k] += x[n] * c * np.cos(((np.pi)/N) * (n + 0.5) * k )
            X[k] += x[n] * c  # Equation of DCT
            
    return X

# Define Main function 
def main(): 

    k = np.arange(N)
    
    X_f = DFT_Function(x) # calling DFT function
    X_c = DCT_Function(x) # calling DCT function 
    
    # For pretty looking fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "cm"
    
    # Plot of DFT and DCT 
    plt.figure(figsize=(12,7))
    plt.subplot(2,1,1) 
    plt.stem(np.abs(X_f), label='X_f')
    plt.title("DFT Magnitude")
    plt.xlabel(r'k-th basis (Freq in Hz)')
    plt.ylabel(r'|X_f [k]| (a.u.)')
    plt.title('Magnitude of the DFT of x[n]')
    plt.legend(title='$Magnitude-of-DFT$')

    plt.figure(figsize=(12,7))
    plt.subplot(2,1,2) 
    plt.stem(np.abs(X_c), label='X_c')
    plt.title("DCT Magnitude")
    plt.xlabel(r'k-th basis(Freq in Hz)')
    plt.ylabel(r'|X_c [k]| (a.u.)')
    plt.title('Magnitude of the DCT of x[n]')
    plt.legend(title='$Magnitude-of-DCT$')

    plt.show()
    
    # Verification of Parsevals theorem 
    Sq_DFT = round((np.sum(abs(X_f)**2) ),2) 
    Sq_DCT = round((np.sum(abs(X_c)**2)),2) 
    
    if Sq_DFT == Sq_DCT:
        print ('Comments on the proof of Parseval Theorem: The Parseval-s theorem is found: True!')
    else:
        print ('Comments on the proof of Parseval Theorem: The parseval-s theorem is found: False!' )
    
    
if __name__ == '__main__':
    main()
