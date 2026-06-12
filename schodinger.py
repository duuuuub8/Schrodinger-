import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit
from scipy.special import eval_hermite

N = 1000
L = 0.000000001
hbar = ((6.63 * (10 ** (-34))) / (2 * np.pi))
m = (9.11 * (10 ** -31))

def ipwell():
    nw = np.linspace(1 , N , N)
    xw = np.linspace(0 , L , N + 1)
    hw = xw[1] - xw[0]
    
    def well(N):
        diag = np.zeros((N + 1 , N + 1))
        for i in range(N + 1):
            diag[i , i] = -2 * ((-hbar ** 2) / (2 * m * (hw ** 2)))
            
        for i in range(N):
            diag[i + 1 , i] = 1 * ((-hbar ** 2) / (2 * m * (hw ** 2)))
            
        for i in range(N):
            diag[i , i + 1] = 1 * ((-hbar ** 2) / (2 * m * (hw ** 2)))
        
        d = np.zeros(N + 1)
        e = np.zeros(N) 
        
        for i in range(N + 1):
            d[i] = diag[i , i]
    
        for i in range(N):
            e[i] = diag[i + 1 , i]
        
        E , v = eigh_tridiagonal(d , e)
        
        return E , v
    
    def wavefunc(N , nw):
        psi = np.zeros(N + 1)
        
        for i in range(N + 1):
            psi[i] = np.sqrt(2 / L) * np.sin((nw * np.pi * xw[i]) / L)
    
        return psi
    
    def wellsol(nw):
        sol = np.zeros(5)
        
        for i in range(5):
            sol[i] = (((nw[i] ** 2) * (np.pi ** 2) * (hbar ** 2)) / (2 * m * (L ** 2)))
            
        return sol
    
    def quadraticfit(x , a):
        
        y =  (a * (x ** 2))
        
        return y
    
    def functionacc(psin ,  psi , a):
        tot = 0
        j = 0
        
        if a == 0:
            for i in range(N):
                if psi[i] != 0:
    #                print(psin[i] , "  " , psi[i + a] , "  " , i)
                    tot = tot + (100 - (((abs(abs(psi[i + a]) - abs(psin[i]))) / psin[i]) * 100))
                    j = j + 1
                else:
                        tot = tot + 0
        elif a == 1:
            for i in range(N):
                if psi[i] != 0:
    #                print(psin[i] , "  " , psi[i + a] , "  " , i)
                    tot = tot + (100 - (((abs(abs(psin[i]) - abs(psi[i + a]))) / psi[i + 0]) * 100))
                    j = j + 1
                else:
                        tot = tot + 0
        acc = tot / (j + 1)
        return acc
    
    Ew = well(N)[0]

    Ewn = Ew / Ew[0]
    Ewnpl = np.array([Ewn[0] , Ewn[1] , Ewn[2] , Ewn[3] , Ewn[4]])
    npl = np.array([nw[0] , nw[1] , nw[2] , nw[3] , nw[4]])
    Ewc = np.array([Ew[0] , Ew[1] , Ew[2] , Ew[3] , Ew[4]])

    realsolw = wellsol(nw)

    unnormpsi = well(N)[1]

    psiun1 = unnormpsi[: , 0]
    psiun2 = unnormpsi[: , 1]
    psiun3 = unnormpsi[: , 2]
    psiun4 = unnormpsi[: , 3]
    psiun5 = unnormpsi[: , 4]

    psi1 = wavefunc(N , nw[0])
    psi2 = wavefunc(N , nw[1])
    psi3 = wavefunc(N , nw[2])
    psi4 = wavefunc(N , nw[3])
    psi5 = wavefunc(N , nw[4])

    psin1 = psiun1 / np.sqrt(np.sum(np.abs(psiun1) ** 2) * hw)
    psin2 = psiun2 / np.sqrt(np.sum(np.abs(psiun2) ** 2) * hw)
    psin3 = psiun3 / np.sqrt(np.sum(np.abs(psiun3) ** 2) * hw)
    psin4 = psiun4 / np.sqrt(np.sum(np.abs(psiun4) ** 2) * hw)
    psin5 = psiun5 / np.sqrt(np.sum(np.abs(psiun5) ** 2) * hw)

    psin1 = np.flip(psin1)
    psin2 = np.flip(psin2)
    psin3 = np.flip(psin3)
    psin4 = np.flip(psin4)
    psin5 = np.flip(psin5)

    accuracyw = np.zeros(5)

    for i in range(5):
        accuracyw[i] = Ewc[i] / realsolw[i]

    bestfitp , seq = curve_fit(quadraticfit , npl , Ewnpl)

    xpoints = np.linspace(0 , len(npl) , 1000)
    ypoints = quadraticfit(xpoints , *bestfitp)

    resq = Ewnpl - quadraticfit(npl , *bestfitp)
    ss_resq = np.sum(resq ** 2)

    ss_totq = np.sum((Ewnpl - np.mean(Ewnpl)) ** 2)

    RRq = 1 - (ss_resq / ss_totq)
    
    facc = np.array([functionacc(psin1 , psi1 , 1) , functionacc(psin2 , psi2 , 0) , functionacc(psin3 , psi3 , 1) , functionacc(psin4 , psi4 , 0) , functionacc(psin5 , psi5 , 0)])
    
    avfacc = np.sum(facc) / 5
    
    plt.scatter(npl , Ewnpl , s = 15 , marker = "x")
    plt.plot(xpoints , ypoints)
    plt.xlabel("n (Dimentionless)")
    plt.ylabel("normalised E (Dimentionless)")
    plt.savefig("pot_well_wave_func_E.png" , bbox_inches = "tight")
    plt.show()


    plt.plot(xw , psi1 , label = "n = 1")
    plt.plot(xw , psi2 , label = "n = 2")
    plt.plot(xw , psi3 , label = "n = 3")
    plt.plot(xw , psi4 , label = "n = 4")
    plt.plot(xw , psi5 , label = "n = 5")
    plt.xlabel("x (m)")
    plt.ylabel("wave function ψ (m^0.5)")
    plt.legend()
    plt.savefig("pot_well_wave_func_real.png" , bbox_inches = "tight")
    plt.show()


    plt.plot(xw , psin1 , label = "n = 1")
    plt.plot(xw , psin2 , label = "n = 2")
    plt.plot(xw , psin3 , label = "n = 3")
    plt.plot(xw , psin4 , label = "n = 4")
    plt.plot(xw , psin5 , label = "n = 5")
    plt.xlabel("x (m)")
    plt.ylabel("wave function ψ (m^0.5)")
    plt.legend()
    plt.savefig("pot_well_wave_func.png" , bbox_inches = "tight")
    plt.show()
    
    #print(well(N))
    #print(xw)
    #print(Ew)
    #print(Ewn)
    #print(hw)
    #print(nw)
    #print(psi)
    #print(Ewnpl)
    #print(npl)
    #print(psi1)
    #print(psin1)
    #print(psi1)
    #print(len(psi1))
    #print(len(psiun1))
    #print(np.sum(np.abs(psin1) ** 2) * hw)
    #print(functionacc(psin2 , psi2 , 0))
    
    print(realsolw)
    print(Ewc)
    print(accuracyw)
    print(RRq)
    print(facc)
    print(avfacc)

def oscillater():
    k = 10
    w = np.sqrt(k / m)
    xh = np.linspace(-L , L , (N + 1))
    hh = xh[1] - xh[0]
    nh = np.linspace(0 , N , N + 1)
    
    def harmonic(N):
        diag = np.zeros((N + 1 , N + 1))
        
        for i in range(N + 1):
            diag[i , i] = (-2 * (((-(hbar ** 2)) / (2 * m * (hh ** 2))))) + (0.5 * k * (xh[i] ** 2))
            
        for i in range(N):
            diag[i + 1 , i] = 1 * ((-(hbar ** 2)) / (2 * m * (hh ** 2)))
            
        for i in range(N):
            diag[i , i + 1] = 1 * ((-(hbar ** 2)) / (2 * m * (hh ** 2)))
        
        d = np.zeros(N + 1)
        e = np.zeros(N) 
        
        for i in range(N + 1):
            d[i] = diag[i , i]
    
        for i in range(N):
            e[i] = diag[i + 1 , i]
        
        E , v = eigh_tridiagonal(d , e)
        
        return E , v
    
    def hermite(N , nh , xh):
        H = np.zeros((5 , N + 1))
    
        for i in range(5):
            n = nh[i]
            for j in range(N + 1):
                H[i] = eval_hermite(int(n) , (np.sqrt((m * w) / hbar)) * xh)      
        return H
    
    def wavefunc(N , nh , H):
        psi = np.zeros(N + 1)
        
        for i in range(N + 1):
            psi[i] = ((((m * w) / (np.pi * hbar)) ** 0.25) * (1 / np.sqrt((2 ** nh) * (math.factorial(int(nh))))) * H[i] * np.exp(-((m * w * (xh[i] ** 2)) / (2 * hbar))))
    
        return psi
    
    def harmsol(nh):
        sol = np.zeros(5)
        
        for i in range(5):
            sol[i] = ((nh[i] + 0.5) * hbar * np.sqrt(k / m))
            
        return sol
    
    def fit(x , a):
        
        y =  (a * x) + 1
        
        return y
    
    def functionacc(psin ,  psi):
        tot = 0
        for i in range(N + 1):
            if psi[i] != 0:
    #            print(psin[i] , "  " , psi[i] , "  " , i)
                tot = tot + (1 - (((abs(abs(psin[i]) - abs(psi[i]))) / psi[i]) ))
            else:
                tot = tot + 0
        acc = tot / (i + 1)
        return acc
    
    Eh = harmonic(N)[0]
    #Eh = np.insert(Eh, 0, (Eh[0] - (hbar * np.sqrt(k / m))) , axis=0)
    realsolh = harmsol(nh)
    
    Ehn = Eh / Eh[0]
    Ehnpl = np.array([Ehn[0] , Ehn[1] , Ehn[2] , Ehn[3] , Ehn[4]])
    npl = np.array([nh[0] , nh[1] , nh[2] , nh[3] , nh[4]])
    Ehc = np.array([Eh[0] , Eh[1] , Eh[2] , Eh[3] , Eh[4]])
    
    H = hermite(N , nh , xh)
    
    unnormpsi = harmonic(N)[1]
    
    psiun0 = unnormpsi[: , 0]
    psiun1 = unnormpsi[: , 1]
    psiun2 = unnormpsi[: , 2]
    psiun3 = unnormpsi[: , 3]
    psiun4 = unnormpsi[: , 4]
    
    psi0 = wavefunc(N , nh[0] , H[0])
    psi1 = wavefunc(N , nh[1] , H[1])
    psi2 = wavefunc(N , nh[2] , H[2])
    psi3 = wavefunc(N , nh[3] , H[3])
    psi4 = wavefunc(N , nh[4] , H[4])
    
    psin0 = psiun0 / np.sqrt(np.sum(np.abs(psiun0) ** 2) * hh)
    psin1 = psiun1 / np.sqrt(np.sum(np.abs(psiun1) ** 2) * hh)
    psin2 = psiun2 / np.sqrt(np.sum(np.abs(psiun2) ** 2) * hh)
    psin3 = psiun3 / np.sqrt(np.sum(np.abs(psiun3) ** 2) * hh)
    psin4 = psiun4 / np.sqrt(np.sum(np.abs(psiun4) ** 2) * hh)
    
    accuracyh = np.zeros(5)
    for i in range(5):
        accuracyh[i] = Ehc[i] / realsolh[i]
    
    bestfitp , seq = curve_fit(fit , npl , Ehnpl)
    
    xpoints = np.linspace(0 , len(npl) - 1 , 1000)
    ypoints = fit(xpoints , *bestfitp)
    
    resq = Ehnpl - fit(npl , *bestfitp)
    ss_resq = np.sum(resq ** 2)
    
    ss_totq = np.sum((Ehnpl - np.mean(Ehnpl)) ** 2)
    
    RRq = 1 - (ss_resq / ss_totq)
    
    facc = np.array([functionacc(psin0 , psi0) , functionacc(psin1 , psi1) , functionacc(psin2 , psi2) , functionacc(psin3 , psi3) , functionacc(psin4 , psi4)])
    
    avfacc = np.sum(facc) / 5
    
    #print(xh)
    #print(Eh)
    #print(Ewn)
    #print(hw)
    #print(nh)
    #print(psi)
    #print(Ehnpl)
    #print(npl)
    #print(psi1)
    #print(psi1)
    #print(H)
    #print(np.sum(np.abs(psin1) ** 2) * hw)
    #print(functionacc(psin2 , psi2))
    
    print(realsolh , "\n")
    print(Ehc , "\n")
    print(accuracyh , "\n")
    print(RRq , "\n")
    print(facc)
    print(avfacc)
    
    plt.scatter(npl , Ehnpl , s = 15 , marker = "x")
    plt.plot(xpoints , ypoints)
    plt.xlabel("n (Dimentionless)")
    plt.ylabel("normalised E (Dimentionless)")
    plt.savefig("harmonic_func_E.png" , bbox_inches = "tight")
    plt.show()
    
    plt.plot(xh , psi0 , label = "n = 0")
    plt.plot(xh , psi1 , label = "n = 1")
    plt.plot(xh , psi2 , label = "n = 2")
    plt.plot(xh , psi3 , label = "n = 3")
    plt.plot(xh , psi4 , label = "n = 4")
    plt.xlabel("x (m)")
    plt.ylabel("wave function ψ (m^0.5)")
    plt.legend()
    plt.savefig("harmonic_func_real.png" , bbox_inches = "tight")
    plt.show()
    
    plt.plot(xh , psin0 , label = "n = 0")
    plt.plot(xh , psin1 , label = "n = 1")
    plt.plot(xh , psin2 , label = "n = 2")
    plt.plot(xh , psin3 , label = "n = 3")
    plt.plot(xh , psin4 , label = "n = 4")
    plt.xlabel("x (m)")
    plt.ylabel("wave function ψ (m^0.5)")
    plt.legend()
    plt.savefig("harmonic_func.png" , bbox_inches = "tight")
    plt.show()

def pertabation():
    k = 10
    w = np.sqrt(k / m)
    xh = np.linspace(-L , L , (N + 1))
    hh = xh[1] - xh[0]
    nh = np.linspace(0 , N , N + 1)
    
    a = int(input("plese imput the power of the x pertabation (a) you are doing (1, 3 or 4): "))
    c = float(input("please imput a coefficient (c) for the pertabation V(x) + cx^a: "))
    #for a = 1: c aprox 0.000000002, for a = 3: c aprox 5000000000, for a = 4: c aprox 20000000000000000000
    
    print(c)
    def pert(N):
        diag = np.zeros((N + 1 , N + 1))
        
        for i in range(N + 1):
            diag[i , i] = (-2 * (((-(hbar ** 2)) / (2 * m * (hh ** 2))))) + (0.5 * k * (xh[i] ** 2)) + (c * (xh[i] ** a))
            
        for i in range(N):
            diag[i + 1 , i] = 1 * ((-(hbar ** 2)) / (2 * m * (hh ** 2)))
            
        for i in range(N):
            diag[i , i + 1] = 1 * ((-(hbar ** 2)) / (2 * m * (hh ** 2)))
        
        d = np.zeros(N + 1)
        e = np.zeros(N) 
        
        for i in range(N + 1):
            d[i] = diag[i , i]
    
        for i in range(N):
            e[i] = diag[i + 1 , i]
        
        E , v = eigh_tridiagonal(d , e)
        
        return E , v
    
    def harmsol(nh , a):
        sol = np.zeros(5)
        
        if a == 1:
            for i in range(5):
                sol[i] = ((nh[i] + 0.5) * hbar * w) - ((c ** 2) / (2 * m * (w ** 2)))
        
        elif a == 3:
            for i in range(5):
                sol[i] = ((nh[i] + 0.5) * hbar * w) - ((c ** 2) * (((15 * (nh[i] ** 2)) + (15 * nh[i]) + 11) / (8 * (m ** 4) * (w ** 4))) * (hbar ** 3))
        
        else:
            for i in range(5):
                sol[i] = ((nh[i] + 0.5) * hbar * w) + (((3 * c * (hbar ** 2)) / (4 * (m ** 2) * (w ** 2))) * ((2 * (nh[i] ** 2)) + (2 * nh[i]) + 1))
        
        return sol
    
    def fit(x , a , b):
        
        y =  (a * x) + b
    #    print(a)
        
        return y
    
    Eh = abs(pert(N)[0])
    #Eh = np.insert(Eh, 0, (Eh[0] - (hbar * np.sqrt(k / m))) , axis=0)
    realsolh = abs(harmsol(nh , a))
    
    Ehn = Eh / Eh[0]
    Ehnpl = np.array([Ehn[0] , Ehn[1] , Ehn[2] , Ehn[3] , Ehn[4]])
    npl = np.array([nh[0] , nh[1] , nh[2] , nh[3] , nh[4]])
    Ehc = np.array([Eh[0] , Eh[1] , Eh[2] , Eh[3] , Eh[4]])
    
    unnormpsi = pert(N)[1]
    
    psiun0 = unnormpsi[: , 0]
    psiun1 = unnormpsi[: , 1]
    psiun2 = unnormpsi[: , 2]
    psiun3 = unnormpsi[: , 3]
    psiun4 = unnormpsi[: , 4]
    
    psin0 = psiun0 / np.sqrt(np.sum(np.abs(psiun0) ** 2) * hh)
    psin1 = psiun1 / np.sqrt(np.sum(np.abs(psiun1) ** 2) * hh)
    psin2 = psiun2 / np.sqrt(np.sum(np.abs(psiun2) ** 2) * hh)
    psin3 = psiun3 / np.sqrt(np.sum(np.abs(psiun3) ** 2) * hh)
    psin4 = psiun4 / np.sqrt(np.sum(np.abs(psiun4) ** 2) * hh)
    
    #psin0 = np.flip(psin0)
    #psin1 = np.flip(psin1)
    #psin2 = np.flip(psin2)
    #psin3 = np.flip(psin3)
    #psin4 = np.flip(psin4)
    
    
    accuracyh = np.zeros(5)
    for i in range(5):
        accuracyh[i] = 1 - abs((Ehc[i] - realsolh[i]) / realsolh[i])
    
    bestfitp , seq = curve_fit(fit , npl , Ehnpl)
    
    xpoints = np.linspace(0 , len(npl) - 1 , 1000)
    ypoints = fit(xpoints , *bestfitp)
    
    resq = Ehnpl - fit(npl , *bestfitp)
    ss_resq = np.sum(resq ** 2)
    
    ss_totq = np.sum((Ehnpl - np.mean(Ehnpl)) ** 2)
    
    RRq = 1 - (ss_resq / ss_totq)
    
    
    #print(xh)
    #print(Eh)
    #print(Ewn)
    #print(hw)
    #print(nh)
    #print(psi)
    print(Ehnpl)
    print(npl)
    #print(psi1)
    #print(psin5)
    #print(psi2)
    #print(H)
    #print(np.sum(np.abs(psin1) ** 2) * hw)
    
    print(realsolh)
    print(Ehc)
    print(accuracyh)
    print(RRq)
    
    
    plt.scatter(npl , Ehnpl , s = 15 , marker = "x")
    plt.plot(xpoints , ypoints)
    plt.xlabel("n (Dimentionless)")
    plt.ylabel("normalised E (Dimentionless)")
    plt.savefig("pert_func_E.png" , bbox_inches = "tight")
    plt.show()
    
    plt.plot(xh , psin0 , label = "n = 0")
    plt.plot(xh , psin1 , label = "n = 1")
    plt.plot(xh , psin2 , label = "n = 2")
    plt.plot(xh , psin3 , label = "n = 3")
    plt.plot(xh , psin4 , label = "n = 4")
    plt.xlabel("x (m)")
    plt.ylabel("wave function ψ (m^0.5)")
    plt.legend()
    plt.savefig("pert_func.png" , bbox_inches = "tight")
    plt.show()

#ipwell()

#oscillater()

pertabation()



