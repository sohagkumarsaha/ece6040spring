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
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
    return X

# Define Function for Discrete Consine Transform 
def DCT(x):
    X = np.zeros(N)
    for k in range(N):
        if k == 0:
            c = 1 / np.sqrt(N)
        else:
            c = np.sqrt(2 / N)
        for n in range(N):
            X[k] += c * x[n] * np.cos(np.pi * (n + 0.5) * k / N)
    return X

# Define Main function 
def main(): 

    
    X_f = DFT(x) # calling DFT function
    X_c = DCT(x) # calling DCT function 

    # Plot of DFT and DCT 
    plt.figure()
    plt.stem(np.abs(X_f), label='X_f')
    plt.title("DFT Magnitude")
    plt.xlabel('k')
    plt.ylabel('|X_f [k]|')
    plt.title('Magnitude of the DFT of x[n]')
    plt.legend(title='$Magnitude-of-DFT$')

    plt.figure()
    plt.stem(np.abs(X_c), label='X_c')
    plt.title("DCT Magnitude")
    plt.xlabel('k')
    plt.ylabel('|X_c [k]|')
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
