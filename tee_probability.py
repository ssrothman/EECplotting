import numpy as np

def Ptheta(theta):
    return 1/(theta)

def P_tee(R, r, phi):
    return Ptheta(R) * Ptheta(r)

def P_wrong1(R, r, phi):
    q2 = R*R + (r/2)*(r/2) - 2*R*(r/2)*np.cos(phi)
    q = np.sqrt(q2)
    return Ptheta(q) * Ptheta(r/2)

def P_wrong2(R, r, phi):
    w2 = R*R + (r/2)*(r/2) - 2*R*(r/2)*np.cos(np.pi-phi)
    w = np.sqrt(w2)
    return Ptheta(w) * Ptheta(r/2)

def Pright(R, r, phi):
    tee = P_tee(R, r, phi)
    wrong1 = P_wrong1(R, r, phi)
    wrong2 = P_wrong2(R, r, phi)
    return tee / (tee + wrong1 + wrong2)

def Pwrong1(R, r, phi):
    tee = P_tee(R, r, phi)
    wrong1 = P_wrong1(R, r, phi)
    wrong2 = P_wrong2(R, r, phi)
    return wrong1 / (tee + wrong1 + wrong2)

def Pwrong2(R, r, phi):
    tee = P_tee(R, r, phi)
    wrong1 = P_wrong1(R, r, phi)
    wrong2 = P_wrong2(R, r, phi)
    return wrong2 / (tee + wrong1 + wrong2)

#import matplotlib.pyplot as plt
#R = 0.35
#r = np.linspace(0.01, 1, 100)
#
#plt.plot(r, Pright(R, r*R, 0), label="phi=0")
#plt.plot(r, Pright(R, r, np.pi/4), label="phi=pi/4")
#plt.plot(r, Pright(R, r, np.pi/2), label="phi=pi/2")
#plt.xlabel("r/R")
#plt.ylabel("Probability the tee is the 'true' configuration")
#plt.legend()
#plt.show()
#
#R = 0.35
#phi = np.linspace(0.0, np.pi*2, 100)
#
#plt.plot(phi, Pright(R, 0.1*R, phi), label="r=0.1")
#plt.plot(phi, Pright(R, 0.3*R, phi), label="r=0.3")
#plt.plot(phi, Pright(R, 0.5*R, phi), label="r=0.5")
#plt.xlabel("phi")
#plt.ylabel("Probability the tee is the 'true' configuration")
#plt.legend()
#plt.show()
