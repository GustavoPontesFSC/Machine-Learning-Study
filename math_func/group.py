import numpy as np

import os
from spectra.classes import libsmean

def y_ml(A): 
    total_shape = 0
    for i in A:
        total_shape +=i
    
    y = np.zeros(total_shape)

    groups = len(A)
    k=0
    for i in range(groups):
        for j in range(A[i]):
            y[k] = i
            k+=1
    return y

def dirtoread(all_sample, DIR, matrix, spectra,HEADER = 41,DELIM=';', libsmean=libsmean): #Faz a leitura de todas as pastas
    for i in all_sample:
        sample = DIR.joinpath(i)
        all_piece = sorted(os.listdir(sample))
        for j in all_piece:
            piece = sample.joinpath(j)
            matrix.append(libsmean(piece, spectra=spectra, HEADER=41,DELIM=DELIM))

