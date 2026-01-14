import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

    #wybor metody numerycznej
    metoda = input("Podaj rodzaj metody numerycznej, RK2 lub RK4: ")
    
    #czas symulacji
    krok = np.array([0.5,0.25,0.1,0.05])
    
    #tablica bledow dla kazdego kroku
    E = np.empty(np.size(krok))
    
    for index,h in enumerate(krok):
        t_end = 10
        N = int(t_end/h + 1)
        t = np.linspace(0,t_end,N)

        #rozwiazanie referencyjne
        print(f'liczenie... metoda referencyjna, krok={h}, F={F[21]:.3f}')
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

        if metoda == "RK2":
            x1_num, x2_num = MojRK2(t_end, c, k, omega, F[21], h, position0, velocity0)
        elif metoda == "RK4":
            x1_num, x2_num = MojRK4(t_end, c, k, omega, F[21], h, position0, velocity0)
        else:
            print("Bledna metoda")
            break

        #wykres aktualnego rozwiazania
        plot_x(x1_ref,x1_num,t,h,metoda)
        
        E[index] = np.max(np.abs(x1_ref-x1_num))

    plot_E(E,krok,metoda)


def system(t, stan, c, k, F, omega):
    x1, x2 = stan
    dx1dt = f1(x2)
    dx2dt = f2(x1,x2,t,c,k,omega,F)
    return [dx1dt, dx2dt]

def MojRK2(t_end,c,k,omega,F,h,position0,velocity0):
    print(f'liczenie... RK2, krok={h}, F={F:.3f}')
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
    print(f'liczenie... RK4, krok={h}, F={F:.3f}')
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

#wykres x(t) porownany z rozwiazaniem referencyjnym
def plot_x(x_ref,x_num,t,krok,metoda):
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_ref, label=f'Rozwiązanie referencyjne (solve_ivp)', linestyle='--',color='blue')
    plt.plot(t, x_num, label=f'Rozwiazanie numeryczne', linestyle='-',color='red')
    plt.xlabel('czas [t]')
    plt.ylabel('x(t)')
    plt.legend()

    #dla rozwiazan niestabilnych wartosci na osi pionowej sa zbyt duze
    #dodaję notację wykładniczą na górze wykresu
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Porównanie rozwiązań, metoda {metoda}, krok {krok}')


    opis = 'rozwiazanie_' + metoda + '_' + str(krok) + '.png'
    path = 'wykresy/' + opis
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return 0

# wykres bledu
def plot_E(E,krok,metoda):
    plt.figure(figsize=(10, 6))
    plt.plot(krok, E, label=f'{metoda}: x1', marker='x', linestyle='')
    plt.xlabel('krok')
    plt.ylabel('blad E')

    #skala logarytmiczna
    plt.yscale('log')
    plt.xscale('log')

    #dodaj znaczniki na osi poziomej w każdym kroku
    plt.xticks(krok, [f'{h}' for h in krok])

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'wykres błędu bezwzględnego E od kroku h, metoda {metoda}')

    opis = 'wykres_bledu_' + metoda + ".png"
    path = 'wykresy/' + opis
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return 0

def plot_phaseplane():
    return 0



if __name__ == "__main__":
    main()
