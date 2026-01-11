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

    scipy.integrate.solve_ivp(method='RK45', max_step=1e-4)

    #E = np.max(x_num - x_ref)

    print(F)
    print(krok)

def MojRK2(t,c,k,omega,F,h):

    return 0
def MojRK4(t,c,k,omega,F,h):
    return 0


if __name__ == "__main__":
    main()
