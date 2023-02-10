# Name: Sohag Kumar Saha
# Assignment-2 (Problem 3) 
# ECE 6040 (Signal Analysis), Spring 2023
# Tennessee Tech University 

#import numpy, matplotlib library


# Name: Sohag Kumar Saha
# Homework 2, Problem 3

import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
x = 0.1 * n

# Define Function for Discrete Fourier Transform 
def DFT(x):
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * (1/(np.sqrt(N)))  * np.exp((1j *2* np.pi * k * n) / N)
    return X

# Define Function for Discrete Consine Transform 
def DCT(x):
    X = np.zeros(N)
    for k in range(N):
        if k == 0:
            c = 1 / np.sqrt(N)
        else:
            c = np.sqrt(2 / N) 
            #c = np.sqrt(2 / N) * np.cos(((np.pi)/N) * (n + 0.5) * k )
            
        for n in range(N):
            X[k] += x[n] * c * np.cos(((np.pi)/N) * (n + 0.5) * k )
            #X[k] += x[n] * c 
            
    return X

# Define Main function 
def main(): 

    
    X_f = DFT(x) # calling DFT function
    X_c = DCT(x) # calling DCT function 


    # For pretty looking fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "cm"
    
    # Plot of DFT and DCT 
    plt.figure()
    plt.stem(np.abs(X_f), label='X_f')
    plt.title("DFT Magnitude")
    plt.xlabel(r'k-th basis (Nos.)')
    plt.ylabel(r'|X_f [k]| (a.u.)')
    plt.title('Magnitude of the DFT of x[n]')
    plt.legend(title='$Magnitude-of-DFT$')

    plt.figure()
    plt.stem(np.abs(X_c), label='X_c')
    plt.title("DCT Magnitude")
    plt.xlabel(r'k-th basis(Nos.)')
    plt.ylabel(r'|X_c [k]| (a.u.)')
    plt.title('Magnitude of the DCT of x[n]')
    plt.legend(title='$Magnitude-of-DCT$')

    plt.show()
    
    # Verification of Parsevals theorem 
    Sq_DFT = np.sum(abs(X_f)**2) 
    Sq_DCT = np.sum(abs(X_c)**2)
    
    if Sq_DFT == Sq_DCT:
        print ('The parseval theorem is true!')
    else:
        print ('The parseval theorem is false!' )
    
    
if __name__ == '__main__':
    main()
