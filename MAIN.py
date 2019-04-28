"""
@author: Simon & Gabriel

Official Main
"""

import numpy as np
import pylab as py
import scipy as sc
from functions import *
from Newton import newton_solver

L = 1
T = 1

# Forme initiale
N_pts_ini = 500
theta = np.linspace(0,2*np.pi,N_pts_ini)
x_t,y_t = (L/2/np.pi)*np.cos(theta),(L/2/np.pi)*np.sin(theta)
t = np.linspace(0,1,N_pts_ini)

# Paramètres de simulation
Ne = 100
N_gauss = 10
deg_case = 1 # degré des fonctions de base utilisées, 1 ou 2
F_range = range(15,16)
Fcase = 0 # 0 force uniforme, 1 force gaussienne avec facteur
plotcase = [0,0,0]  # Itération Newton, Solutions esp. lin., Solutions esp. réel  (1 : plot, 0 : no plot)
savecase = [0,0,0] # (1 : save, 0 : no save)
directory = ['misc/','misc/','misc/']  #Les répertoires doivent déja exister sur l'ordinateur


# GO!!!
[EL2,EH1,coeff] = newton_solver(1,1,x_t,y_t,t,40,N_gauss,2,range(10,11),0,plotcase,savecase,directory)




###############################################################################################
# Graphiques de taux de convergence
'''
F = [range(5,6),range(10,11),range(20,21)]
deg = [1,2]
N_elts = [i for i in range(10,200,10)]
Fcase = 0
i=0
figures = []
for F_range in F:
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F = ',F_range[0])
    
    fig = py.figure(i)
    py.title('Force = ' + str(F_range[0]))
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        
    for deg_case in deg:
        print('######### p=',deg_case)
        
        EL,EH = [],[]
        
        for Ne in N_elts:
            print('Ne = ',Ne)
            [EL2,EH1] = newton_solver(L,T,x_t,y_t,t,Ne,N_gauss,deg_case,F_range,Fcase,plotcase,savecase,directory)
            EL.append(EL2)
            EH.append(EH1)
        
        ax1.loglog(N_elts,EL,marker='o',label='p = ' + str(deg_case))
        ax1.set_xlabel('Ne')
        ax1.set_ylabel('Erreur L2')
        ax1.legend(loc = 'upper right')

        ax2.loglog(N_elts,EH,marker='o',label='p = ' + str(deg_case))
        ax2.set_xlabel('Ne')
        ax2.set_ylabel('Erreur H1')
        ax2.legend(loc = 'upper right')
        
        aL2 = abs(np.diff(np.log(np.transpose(EL)[0])))/np.diff(np.log(N_elts))
        aH1 = abs(np.diff(np.log(np.transpose(EH)[0])))/np.diff(np.log(N_elts))
        print('alpha L2 = ',aL2)
        print('alpha H1 = ',aH1)
    
    i+=1
        
    figures.append(fig)
        
figures[0].show()
figures[1].show()
figures[2].show()

'''

'''
###############################################################################################
# Taux de convergence pour la fonctionnelle de la longueur

deg = [2]
N_elts = [i for i in range(10,50,5)]
Fcase = 0
F_range = range(10,11)
py.figure()
for deg_case in deg:
    print('######### p=',deg_case)
    
    deltaL = []
    
    for Ne in N_elts:
        print('Ne = ',Ne)
        [EL2,EH1,dL] = newton_solver(L,T,x_t,y_t,t,Ne,N_gauss,deg_case,F_range,Fcase,plotcase,savecase,directory)
        
        deltaL.append(dL)
        
    py.plot(N_elts,deltaL,marker='o',label = 'p = ' + str(deg_case))
    py.xlabel('Ne')
    py.ylabel(r'$\Delta$L')
    
    i+=1

py.show()
'''
    
      
    
