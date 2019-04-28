import numpy as np
import pylab as py
import scipy as sc


def hello(x):
    print(x*np.pi)

## Fonctions de transformation d'espace
def curve_analysis(x_t,y_t,t):
    
    #radius derivative function
    xp=np.diff(x_t)/np.diff(t)
    xp=np.append(xp,xp[np.size(xp)-1])
    yp=np.diff(y_t)/np.diff(t)
    yp=np.append(yp,yp[np.size(yp)-1])


    #t to coord(input) 
    t_to_x=sc.interpolate.interp1d(t,x_t,kind='quadratic')
    t_to_y=sc.interpolate.interp1d(t,y_t,kind='quadratic')

    #Normal vector
    norm_N=np.sqrt(yp**2+xp**2)
    Nx=yp/norm_N
    Ny=-xp/norm_N
    
    t_to_Nx=sc.interpolate.interp1d(t,Nx,kind='quadratic')
    t_to_Ny=sc.interpolate.interp1d(t,Ny,kind='quadratic')   

    #curvilign absciss
    #define de bijection between S and t
    ##IMPORTANT### For this funcion to work correctly a lot of point t are required.
    S=np.array([0])
    for i in np.arange(2,np.size(t)+1,1):
        inte=sc.integrate.trapz(np.sqrt(xp[0:i]**2+yp[0:i]**2),t[0:i])
        S=np.append(S,inte)


    ##function t_to_S and S_to_t 
    t_to_S=sc.interpolate.interp1d(t,S,kind='quadratic')
    S_to_t=sc.interpolate.interp1d(S,t,kind='quadratic')

    return xp,yp,S,t_to_x,t_to_y,t_to_Nx,t_to_Ny,t_to_S,S_to_t


def Nchap_1(ksi):
    return (1-ksi)/2
def Nchap_2(ksi):
    return (1+ksi)/2
def Nchap_3(ksi):
    return 1-ksi**2
def Nchap_1_p(ksi):
    return (-1/2)*np.ones(np.size(ksi))
def Nchap_2_p(ksi):
    return (1/2)*np.ones(np.size(ksi))
def Nchap_3_p(ksi):
    return -2*ksi
def N_unitaire(x):
    return 1
 
    
def Te(ksi,element,nodes):
    return (ksi+1)/2*(nodes[element+1])+(1-ksi)/2*(nodes[element])


## Special integration by Gauss quadrature    
def special_int(n,element,nodes,fct_forme,deg):   # fct_forme est une liste qui contient les Nchap
    # Gauss points for specified n
    f = open('gauss_points.txt','r')
    a,i,wi,xi,I = int(2+3/2*n+n**2/2),1,[],[],0
    pts = [k for k in range(a,a+n)]
    for line in f.readlines():
        if i in pts:
            l = line.split()
            wi.append(float(l[1]))
            xi.append(float(l[2]))
        i += 1
        
    # Integral evaluation
    h = nodes[element+1]-nodes[element]
    for j in range(len(wi)):
        P = 1
        for f in fct_forme:
            P *= f(xi[j])
        I += P*wi[j]*(2/h)**(sum(deg)-1)
        
    return I   


## Fonction de maillage
def mesh(S,Nb_element):  #uniform for now
    xmin=0
    xmax=S[np.size(S)-1]
    #equidistant
    nodes=np.linspace(xmin,xmax,Nb_element+1)
    #Connectivity matrix NX2
    connect=[[i,i+1] for i in range(Nb_element)]
    
    return nodes,connect

## Connect element to global function
def connectddl(connect,degcase):
     if degcase==1:
         return connect
     elif degcase==2:
         return [connect[i]+[connect[len(connect)-1][1]+i+1] for i in range(len(connect))]
     
     
 
def dirichlet(connect,K,F):
    index1=0
    index2=connect[-1][1]
    
    #replace columns
    for i in K:
        i[np.array([index1,index2])]=0
    ##replace 2 lines
    F[index1]=0
    F[index2]=0
    vec1=np.zeros(K.shape[0])
    vec2=np.zeros(K.shape[0])
    vec1[index1]=1
    vec2[index2]=1
    K[index1]=vec1
    K[index2]=vec2
     
    
    
    
def plot_function(nElement,nodes,coeff,ddl,xlim,ylim,deg_case):
    for i in range(nElement):
        x=np.linspace(nodes[i],nodes[i+1],100)
        ksi=np.linspace(-1,1,100)
        f=coeff[ddl[i][0]]*Nchap_1(ksi)
        f+=coeff[ddl[i][1]]*Nchap_2(ksi)
        if deg_case == 2:
            f+=coeff[ddl[i][2]]*Nchap_3(ksi)
        if i%2 == 0:
            py.plot(x,f,color='r')
        else:
            py.plot(x,f,color='b')
        py.xlim(xlim)
        py.ylim(ylim)

     
def errorH1(nElement,nodes,coeff1,coeff2,ddl,deg_case):
    errorL2,errorH1 = 0,0
    for i in range(nElement):
        h = nodes[i+1]-nodes[i]
        x=np.linspace(nodes[i],nodes[i+1],500)
        ksi=np.linspace(-1,1,500)
        f1=coeff1[ddl[i][0]]*Nchap_1(ksi)
        f1+=coeff1[ddl[i][1]]*Nchap_2(ksi)
        f2=coeff2[ddl[i][0]]*Nchap_1(ksi)
        f2+=coeff2[ddl[i][1]]*Nchap_2(ksi)
        f1p=coeff1[ddl[i][0]]*Nchap_1_p(ksi)*2/h
        f1p+=coeff1[ddl[i][1]]*Nchap_2_p(ksi)*2/h
        f2p=coeff2[ddl[i][0]]*Nchap_1_p(ksi)*2/h
        f2p+=coeff2[ddl[i][1]]*Nchap_2_p(ksi)*2/h
        
        if deg_case == 2:
            f1+=coeff1[ddl[i][2]]*Nchap_3(ksi)
            f2+=coeff2[ddl[i][2]]*Nchap_3(ksi)
            f1p+=coeff1[ddl[i][2]]*Nchap_3_p(ksi)*2/h
            f2p+=coeff2[ddl[i][2]]*Nchap_3_p(ksi)*2/h
        
        errorL2+=sc.integrate.trapz((f1-f2)**2,x)
        errorH1+=sc.integrate.trapz((f1-f2)**2+(f1p-f2p)**2,x)
        
    return np.sqrt(errorL2),np.sqrt(errorH1)    


def solexacte(x,a,L):
    return a*np.cosh(x/a-L/(2*a))-a*np.cosh(L/(2*a))
def solexactepp(x,a,L):
    return np.sinh(x/a-L/(2*a))


def error_final(nElement,nodes,cddl,deg_case,u,a,L):
    errorL2,errorH1=0,0
    
    for i in range(nElement):
        h = nodes[i+1]-nodes[i]
        x=np.linspace(nodes[i],nodes[i+1],500)
        ksi=np.linspace(-1,1,500)
        f1=u[cddl[i][0]]*Nchap_1(ksi)
        f1+=u[cddl[i][1]]*Nchap_2(ksi)
        f1p=u[cddl[i][0]]*Nchap_1_p(ksi)*2/h
        f1p+=u[cddl[i][1]]*Nchap_2_p(ksi)*2/h
        
        if deg_case == 2:
            f1+=u[cddl[i][2]]*Nchap_3(ksi)
            f1p+=u[cddl[i][2]]*Nchap_3_p(ksi)*2/h
        
        f2=solexacte(x,a,L)
        f2p=solexactepp(x,a,L)
        
        errorL2+=sc.integrate.trapz((f1-f2)**2,x)
        errorH1+=sc.integrate.trapz((f1-f2)**2+(f1p-f2p)**2,x)
        
#        py.figure(1)
#        py.plot(x,f1)
#        py.plot(x,f2,color='k')
#        py.show()
#        
#        py.figure(2)
#        py.plot(x,f1p)
#        py.plot(x,f2p,color='k')
#        py.show()
        
    return np.sqrt(errorL2),np.sqrt(errorH1)   
    


def S_to_u(S,coeff,nElement,nodes,ddl,deg_case):
    u=np.zeros(np.size(S))
    for i in range(nElement):
        index=np.where((nodes[i]<=S) & (S<nodes[i+1]))
        S_el=S[index]
        h=nodes[i+1]-nodes[i]
        ksi=2/h*(S_el-(nodes[i+1]+nodes[i])/2)
        f=coeff[ddl[i][0]]*Nchap_1(ksi)
        f+=coeff[ddl[i][1]]*Nchap_2(ksi)
        if deg_case == 2:
            f+=coeff[ddl[i][2]]*Nchap_3(ksi)
        u[index]=f
    return u

def longueur(coeff,nodes,Nelement,ddl,deg_case):
    longueur=0
    for i in range(Nelement):
        h=nodes[i+1]-nodes[i]
        ksi=np.linspace(-1,1,100)
        shape=coeff[ddl[i][0]]*Nchap_1_p(ksi)
        shape+=coeff[ddl[i][1]]*Nchap_2_p(ksi)
        if deg_case==2:
            shape+=coeff[ddl[i][2]]*Nchap_3_p(ksi)
        dL=np.sqrt(1+4/h**2*shape**2)
        longueur+=h/2*sc.integrate.trapz(dL,ksi)
    return longueur

def longueurAna(funp,S,a,L):
    return sc.integrate.trapz(np.sqrt(1+funp(S,a,L)**2),S)



