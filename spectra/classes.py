
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from math_func.fuctions import Lorentzian

        
class spectra(object):
    def __init__(self, path, HEADER=41, DELIM=';', W=0, I=2, ON_SNV = False):
        """
        Start spectrum with intensity and wavelength
        """
        self.W = W
        self.I = I
        try:
            self.matrix = pd.read_csv(path, sep=DELIM, skiprows=HEADER, header=None).sort_values(0).to_numpy()

        except UnicodeDecodeError:
            self.matrix = pd.read_csv(path, sep=DELIM, skiprows=HEADER, header=None, encoding='iso-8859-1').sort_values(0).to_numpy()

        finally:
            self.wavelengths, self.intensity = self.matrix[:,self.W], self.matrix[:,self.I]
            if ON_SNV:  
                self.ON_SNV = ON_SNV
                I_m = np.mean(self.intensity, axis=0)
                self.intensity = (self.intensity - I_m)/np.std(self.intensity)
            else:
                self.ON_SNV = False # If False, it is possible to use the SNV function 
            self.spectras = len(self.matrix[0,1:])
    def lim_w(self, points):
        """
        Return a fuction limited within given points
        """
        return (self.wavelengths > points[0]) & (self.wavelengths < points[1])
    def analitical_curve(self,points, N=500, lim_w=lim_w):
        '''
        This fuction return a peak area analitical, but not used in this programm
        '''
        
        if N <2:
            raise ValueError("N can't be less than 2")
        mask = lim_w(self, points=points)
        x = np.linspace(*points, N) # column x magnified with N points
        y = np.interp(x, self.wavelengths[mask], self.intensity[mask]) # interpolate of the intensity from aproximade estimate
        arg = np.argmax(y)
        yc = y[arg]
        xc = x[arg]
        y0 = 0
        ind = np.argmin(np.abs(yc/2-self.intensity[mask]))
        w = 2*(xc-x[ind])
        A = np.trapz(self.intensity[mask], self.wavelengths[mask])
        aprox = (A, w, xc, y0) # approx values

        try:
            fit = curve_fit(Lorentzian, self.wavelengths[mask], self.intensity[mask], p0= aprox)
        except RuntimeError as erro:
            raise RuntimeError('Your approx need better variables!')
        return Lorentzian(self.wavelengths[mask], *fit[0])

class spectra_SNV(spectra):
       def __init__(self, path, HEADER=41, DELIM=';', W=0, I=2, ON_SNV=True):
           super().__init__(path, HEADER, DELIM, W, I, ON_SNV)

#Definitions

def libsmean(DIR_Pr, spectra, HEADER, DELIM = ';'):
    print(DIR_Pr)
    A = spectra(DIR_Pr/'_ (1).esf', HEADER=HEADER, DELIM=DELIM)
    M_SNV = np.zeros(np.shape(A.intensity)[0])
    all_shot = sorted(os.listdir(DIR_Pr))
    for i in all_shot:
        shot = DIR_Pr.joinpath(i)
        A = spectra(shot)
        M_SNV += A.intensity
    M_SNV /= len(all_shot)  
    return M_SNV

