import numpy as np
import scipy
import matplotlib.pyplot as plt

def main():

    #parametry rownania
    c = 1.0
    k = 25
    omega = 10

    #rozwiazujemy rownanie dla wielu F0 z zakresu [0,1000]
    #tworzenie wektora F ze zmienna gestoscia w kolejnych przedzialach
    F1 = np.linspace(0, 1, 10)
    F2 = np.linspace(1, 30, 50)
    F3 = np.linspace(30, 100, 20)
    F4 = np.linspace(100, 1000, 10)
    F = np.concatenate((F1, F2, F3, F4))

    #warunki poczatkowe
    position0 = 0
    velocity0 = 0
    
    #czas symulacji
    krok = np.array([0.5,0.25,0.1,0.05])
    t_end = 10
    N = int(t_end/krok[2] + 1)
    t = np.linspace(0,t_end,N)

    x_ref = scipy.integrate.solve_ivp(
        system,
        (0,t_end),
        [position0,velocity0],
        args=(c,k,F[21],omega),
        t_eval = t,
        method='RK45',
        max_step=1e-4
    )

    x1_ref = x_ref.y[0]
    x2_ref = x_ref.y[1]

    print(x1_ref)

    x1_num,x2_num = MojRK4(t_end,c,k,omega,F[21],krok[2],position0,velocity0)

    print(x1_num)

def system(t, stan, c, k, F, omega):
    x1, x2 = stan
    dx1dt = f1(x2)
    dx2dt = f2(x1,x2,t,c,k,omega,F)
    return [dx1dt, dx2dt]

def MojRK2(t_end,c,k,omega,F,h,position0,velocity0):
    N = int(t_end/h + 1)
    t = np.linspace(0,t_end,N)

    # inicjalizacja rownan
    x1 = np.empty(N)
    x2 = np.empty(N)
    x1[0] = position0
    x2[0] = velocity0


    for i in range(1,N):
        k1_1 = h*f1(x2[i-1])
        k2_1 = h*f2(x1[i-1],x2[i-1],t[i-1],c,k,omega,F)
        k1_2 = h*f1(x2[i-1] + k2_1)
        k2_2 = h*f2(x1[i-1] + k1_1 ,x2[i-1] + k2_1 ,t[i-1]+h,c,k,omega,F)
        x1[i] = x1[i-1] + 0.5*(k1_1 + k1_2)
        x2[i] = x2[i-1] + 0.5*(k2_1 + k2_2)

    return x1,x2


def MojRK4(t_end,c,k,omega,F,h,position0,velocity0):
    N = int(t_end/h + 1)
    t = np.linspace(0,t_end,N)

    # inicjalizacja rownan
    x1 = np.empty(N)
    x2 = np.empty(N)
    x1[0] = position0
    x2[0] = velocity0


    for i in range(1,N):
        k1_1 = h*f1(x2[i-1])
        k2_1 = h*f2(x1[i-1],x2[i-1],t[i-1],c,k,omega,F)
        k1_2 = h*f1(x2[i-1] + 0.5*k2_1)
        k2_2 = h*f2(x1[i-1] + 0.5*k1_1 ,x2[i-1] + 0.5*k2_1 ,t[i-1]+0.5*h,c,k,omega,F)
        k1_3 = h*f1(x2[i-1] + 0.5*k2_2)
        k2_3 = h*f2(x1[i-1] + 0.5*k1_2 ,x2[i-1] + 0.5*k2_2 ,t[i-1]+0.5*h,c,k,omega,F)
        k1_4 = h*f1(x2[i-1] + k2_3)
        k2_4 = h*f2(x1[i-1] + k1_3 ,x2[i-1] + k2_3 ,t[i-1]+h,c,k,omega,F)
        x1[i] = x1[i-1] + (1/6)*(k1_1 + 2*k1_2 + 2*k1_3 + k1_4)
        x2[i] = x2[i-1] + (1/6)*(k2_1 + 2*k2_2 + 2*k2_3 + k2_4)

    return x1,x2

def f1(x2):
    return x2

def f2(x1,x2,t,c,k,omega,F):
    return -c*x2 - k*x1 + F*np.sin(omega*t)



if __name__ == "__main__":
    main()
