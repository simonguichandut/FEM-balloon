import numpy as np
import pylab as py
import scipy as sc
from fonctions import *

def newton_solver(Length,Tension,x_t,y_t,t,Ne,N_gauss,deg_case,F_range,Fcase,plotcase,savecase,directory):
    
    py.close('all')
    
    [xp,yp,S,t_to_x,t_to_y,t_to_Nx,t_to_Ny,t_to_S,S_to_t] = curve_analysis(x_t,y_t,t)
       
    [nodes,connect] = mesh(S,Ne)
    cddl = connectddl(connect,deg_case)
    
    if deg_case == 1:
        Fonctions = [[Nchap_1,Nchap_2],[Nchap_1_p,Nchap_2_p]]
    elif deg_case == 2: 
        Fonctions = [[Nchap_1,Nchap_2,Nchap_3],[Nchap_1_p,Nchap_2_p,Nchap_3_p]]
    
    EL2,EH1,dL = [],[],[]
    
    for Force in reversed(F_range):
        
        py.close (1)
        py. close(2)
        py.close(3)
        #print('F = ',Force)
        
        # Constantes
        C = Force/(Length*Tension)
        a = 1/C
    
        u_iter = np.vstack((np.zeros(deg_case*Ne+1),np.zeros(deg_case*Ne+1)))

        ErrH1,compteur = 1,0
        
        while ErrH1>1e-7:
            
            K = np.zeros((deg_case*Ne+1,deg_case*Ne+1))
            F = np.zeros((deg_case*Ne+1,1))
            u = u_iter[-1]
            py.close(2)
            
            for i in range(Ne): # boucle sur les éléments
                
                ## Fonctions ###########################################################################################
                def fct_uk_1(ksi):
                    ukprime = u[cddl[i][0]]*Nchap_1_p(ksi) + u[cddl[i][1]]*Nchap_2_p(ksi)
                    if deg_case == 2 :
                        ukprime += u[cddl[i][2]]*Nchap_3_p(ksi)
                    h = nodes[i+1]-nodes[i]
                    return ukprime*(2/h)
                
                def fct_uk_2(ksi):
                    ukprime = u[cddl[i][0]]*Nchap_1_p(ksi) + u[cddl[i][1]]*Nchap_2_p(ksi)
                    if deg_case == 2:
                        ukprime += u[cddl[i][2]]*Nchap_3_p(ksi)
                    h = nodes[i+1]-nodes[i]
                    return np.sqrt(1+ukprime**2*(2/h)**2)
                
                def fct_uk_3(ksi):
                    ukprime = u[cddl[i][0]]*Nchap_1_p(ksi) + u[cddl[i][1]]*Nchap_2_p(ksi) 
                    if deg_case == 2:
                        ukprime += u[cddl[i][2]]*Nchap_3_p(ksi)
                    h = nodes[i+1]-nodes[i]
                    return ukprime*(2/h)/np.sqrt(1+ukprime**2*(2/h)**2)  
                
                
                if Fcase == 0:
                    
                    def force_function(ksi):
                        return Force
                    
                elif Fcase == 1:
                    
                    def force_non_unif(fmax,S):
                        sigma = np.max(S)/15
                        x0 = np.linspace(min(S),max(S),1000)
                        y0 = fmax*np.exp(-(x0-max(x0)/2)**2/(2*sigma**2))
                        I = sc.integrate.trapz(y0,x0)
                        return fmax*np.exp(-(S-np.max(S)/2)**2/(2*sigma**2))-I/np.max(S)
                    
                    Force_pts = force_non_unif(Force,S)
                    S_to_Force = sc.interpolate.interp1d(S,Force_pts,kind='quadratic')
                    
                    def force_function(ksi):
                        x = Te(ksi,i,nodes)
                        return S_to_Force(x)
                ########################################################################################################
                
                ## Assemblage ##########################################################################################
                for j in range(deg_case+1):        
                    
                    f = special_int(N_gauss,i,nodes,[Fonctions[1][j],fct_uk_1],[1,0]) 
                    f += 1/(Length*Tension)*special_int(N_gauss,i,nodes,[Fonctions[0][j],fct_uk_2,force_function],[0,0]) 
                    
                    F[cddl[i][j]] += -f
                    
                    for k in range(deg_case+1):
                        
                        b = special_int(N_gauss,i,nodes,[Fonctions[1][j],Fonctions[1][k]],[1,1])  
                        b += 1/(Length*Tension)*special_int(N_gauss,i,nodes,[Fonctions[1][j],Fonctions[0][k],fct_uk_3,force_function],[1,0]) 
                        
                        K[cddl[i][k],cddl[i][j]] += b
            
            dirichlet(connect,K,F)
            u2 = np.matmul(np.linalg.inv(K),F)     
            [ErrL2,ErrH1] = errorH1(Ne,nodes,u,u + np.transpose(u2)[0],cddl,deg_case)
            u_iter = np.vstack((u_iter,u+np.transpose(u2)[0]))
            #print(ErrL2,' ; ',ErrH1)


            #### Plot ##################################################################################################
            py.figure(1)
            if Force == F_range[-1]:
                ylim = [min(u_iter[-1])-1,1]
                xlim = [-0.1*Length,Length*1.1]
            plot_function(Ne,nodes,u_iter[-1],cddl,xlim,ylim,deg_case)
            py.xlabel('x')
            py.ylabel('u')
            if savecase[0] == 1:    
                filename = directory[0] + 'Force_' + str(Force) + 'Iteration_' + str(compteur) + '.png'
                py.savefig(filename)
            if plotcase[0] == 0:
                py.close(1)


        py.figure(2)
        plot_function(Ne,nodes,u_iter[-1],cddl,xlim,ylim,deg_case)    
        py.annotate('F='+str(Force),xy=(0.9*xlim[1],0.95*ylim[0]))
        py.xlabel('x')
        py.ylabel('u')
        if savecase[1] == 1:
            filename = directory[1] + 'Lin_Force_' + str(Force) + '.png'
            py.savefig(filename)
        if plotcase[1] == 0:
            py.close(2)   
               
        py.figure(3)
        dU = S_to_u(S,u_iter[-1],Ne,nodes,cddl,deg_case)
        Nx=t_to_Nx(t)
        Ny=t_to_Ny(t) 
        x2_t=x_t-dU*Nx
        y2_t=y_t-dU*Ny
        py.plot(x2_t,y2_t,'k')
        if Force == F_range[-1]:
            xlim2 = [min(x2_t)-2,max(x2_t)+2]
            ylim2 = [min(y2_t)-2,max(y2_t)+2]
        py.xlim(xlim2)
        py.ylim(ylim2)
        py.xlabel('x')
        py.ylabel('y')
        py.annotate('F='+str(Force),xy=(0.8*max(xlim2),0.95*min(ylim2)))
        if savecase[2] == 1:
            filename = directory[2] + 'Real_Force_' + str(Force) + '.png'
            py.savefig(filename)
        if plotcase[2] == 0:       
            py.close(3)

        py.show()
        #################################################################################################################
        
        
        # Erreurs versus solution ana
        [ErrL2,ErrH1] = error_final(Ne,nodes,cddl,deg_case,u_iter[-1],a,Length)
        EL2.append(ErrL2)
        EH1.append(ErrH1)   
        
        
        # Erreur sur la longueur
        Lana = longueurAna(solexactepp,S,a,Length)
        Lsol = longueur(u_iter[-1],nodes,Ne,cddl,deg_case)
        #print(Lana,Lsol)
        dL.append(abs(Lana-Lsol))
    
    return EL2,EH1,dL
    
    
    
    
    
    
    