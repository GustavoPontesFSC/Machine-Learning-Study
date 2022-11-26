from numpy import pi

def Lorentzian(x,A,w,xc,y0):
    return y0+((2*A)/pi)*(w/(4*(x-xc)**2 +w**2))