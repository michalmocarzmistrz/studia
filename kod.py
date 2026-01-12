import numpy as np
import scipy
import matplotlib.pyplot as plt

def main():

    #parametry rownania
    t = 10
    c = 1.0
    k = 25
    omega = 10

    #warunki poczatkowe
    position0 = 0
    velocity0 = 0
    #zmienne parametry
    krok = np.array([0.5,0.25,0.1,0.05])

    #rozwiazujemy rownanie dla wielu F0 z zakresu [0,1000]
    #tworzenie wektora F ze zmienna gestoscia w kolejnych przedzialach
    F1 = np.linspace(0, 1, 10)
    F2 = np.linspace(1, 30, 50)
    F3 = np.linspace(30, 100, 20)
    F4 = np.linspace(100, 1000, 10)
    
    F = np.concatenate((F1, F2, F3, F4))

    #scipy.integrate.solve_ivp(method='RK45', max_step=1e-4)

    #E = np.max(x_num - x_ref)

    np.set_printoptions(suppress=True)
    
    print(F)
    print(krok)
    x,dx_dt = MojRK2(t,c,k,omega,F[21],krok[3],position0,velocity0)

    #print(x)

def MojRK2(t,c,k,omega,F,h,position0,velocity0):
    N = int(t/h + 1)
    t = np.linspace(0,t,N)

    # inicjalizacja rownan
    x1 = np.empty(N)
    x2 = np.empty(N)
    x1[0] = position0
    x2[0] = velocity0


    for i in range(1,N):
        fxy1 = x2[i-1]
        fxy2 = -c*x2[i-1] - k*x1[i-1] + F*np.sin(omega*t[i-1])
        x1E = x1[i-1] + h*fxy1
        x2E = x2[i-1] + h*fxy2
        fxyE1 = x2E
        fxyE2 = -c*x2E - k*x1E + F*np.sin(omega*t[i])
        x1[i] = x1[i-1] + h*0.5*(fxy1 + fxyE1)
        x2[i] = x2[i-1] + h*0.5*(fxy2 + fxyE2)


    print(np.array2string(x1, suppress_small=True))

    return x1,x2


def MojRK4(t,c,k,omega,F,h):
    return 0


if __name__ == "__main__":
    main()
