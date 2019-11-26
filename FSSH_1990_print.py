# Surface Hopping - Tully(1990)
 
import math, cmath
import random
import time
import matplotlib.pyplot as plt 



V      = [[0,0,0], [0,0,0], [0,0,0]]  
der_V  = [[0,0,0], [0,0,0], [0,0,0]] 
E      = [[0,0,0], [0,0,0], [0,0,0]]

F      = [[0],     [0],     [0]    ]

a      = [[0,0,0], [0,0,0], [0,0,0]]  #complex
b      = [[0,0,0], [0,0,0], [0,0,0]]
d      = [[0,0,0], [0,0,0], [0,0,0]]

c_dot  = [0,complex(0,0),complex(0,0)]

###################################################################################################

def main_code():

    ntraj     = 2000
    t_step    = 1     		# dt_step = 1

    mass      = 2.0e3	

    xmin,xmax = -10.0,10.0

    momentums, prob11, prob12, prob22 = [], [], [], []
    for momentum in range(1,31):
       	j=[0,0]
        k=[0,0]
 
        for nt in range(0,ntraj):
            
            x         = -9.9999
            r_dot     = momentum/mass
            nstate    = 1
            cstate    = [0,complex(1,0),complex(0,0)]

            V, der_V  = potential(x)
            F         = force(V, der_V)
            E         = eigenvalues(V)

            position, TE, t, ns = [], [],[], []

            while(x>xmin and x<xmax):

            	cstate    	   	 	= rk4(r_dot, cstate, V, E, t_step)			
            	nstate, r_dot  		= hopping_probability(r_dot,cstate,V, nstate, mass, E, t_step, x)

            	x, r_dot, V, F, E 	= velocity_verlet(x, r_dot, mass,F, t_step, nstate )

            if (nstate==1):
            	if (x<0):
            		j[0]=j[0]+1
            	if (x>0):
            		j[1]=j[1]+1
            else:
            	if (x<0):
            		k[0]=k[0]+1
            	if (x>0):
            		k[1]=k[1]+1            	
        prob_trans_lower =float(j[1]/ntraj)
        prob_refl_lower  =float(j[0]/ntraj)
        prob_trans_upper =float(k[1]/ntraj)

        momentums.append(momentum)
        prob12.append(prob_trans_lower)
        prob11.append(prob_refl_lower)
        prob22.append(prob_trans_upper)

        print("-->",momentum,"**",j,k, "--- %s seconds ---" % ( time.time()-start_time) )


    #Plotting Probabilities
    fig, ax = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0.1})
    
    ax[2].set_ylim(None,1)
    ax[0].plot(momentums,prob12, label = "prob_trans_lower", linestyle = "-", marker = "o")
    ax[1].plot(momentums,prob11, label = "prob_refl_lower",  linestyle = "-", marker = "v")
    ax[2].plot(momentums,prob22, label = "prob_trans_upper", linestyle = "-", marker = "s")

    ax[2].set(xlabel='momentum')	
    ax[0].legend(); ax[1].legend(); ax[2].legend();
    plt.show()
#--------------------------------------------------------------------------------------------------#

def potential(x):

    #Potential and it's derivatives
    A,B,C,D = 0.01e0, 1.6e0, 0.005e0, 1e0

    if (x>0): 
        V[1][1]      =  A*(1-math.exp(-B*x))
        der_V[1][1]  =  A*B*(math.exp(-B*x))
    elif (x<0): 
        V[1][1]      = -A*(1-math.exp(B*x))
        der_V[1][1]  =  A*B*(math.exp(B*x))
    
    V[2][2]          = -V[1][1]
    der_V[2][2]      = -der_V[1][1]
    V[1][2]          = C*(math.exp(-D*x**2))
    der_V[1][2]      = -2*x*C*D*(math.exp(-D*x**2))
    V[2][1]          = V[1][2]
    der_V[2][1]      = der_V[1][2]
    return V, der_V
#--------------------------------------------------------------------------------------------------#

def force(V, der_V):
    
    F[1]= (1/math.sqrt((V[1][1]**2)+(V[1][2]**2))) * ((V[1][1]*der_V[1][1]) + (V[1][2]*der_V[1][2]))
    F[2]=-(1/math.sqrt((V[1][1]**2)+(V[1][2]**2))) * ((V[1][1]*der_V[1][1]) + (V[1][2]*der_V[1][2]))	
    return F
#--------------------------------------------------------------------------------------------------#

def eigenvalues(V):


    E[1][1] = ( (V[1][1]+V[2][2])-math.sqrt((V[1][1]+V[2][2])**2-(4*(V[1][1]*V[2][2]-(V[1][2])**2))) )/2.e0	
    E[2][2] = ( (V[1][1]+V[2][2])+math.sqrt((V[1][1]+V[2][2])**2-(4*(V[1][1]*V[2][2]-(V[1][2])**2))) )/2.e0	
    E[1][2] = 0
    E[2][1] = 0
    return E
#--------------------------------------------------------------------------------------------------#

def equation(r_dot, cstate, V, E):
 
    # Complex valued expansion coefficients c_j(t)  
    """
    Wavefunction Ψ(R,r,t) is expanded in terms of electronic basis functions
      Ψ(r,R,t) = Σ_j c_j(t) * Φ_j(r,R)
    """         
    IOTA   = complex(0,1)
    c_dot[1] = ((E[1][1]/IOTA))*cstate[1] + (-(r_dot*d[1][2]))*cstate[2]
    c_dot[2] = ((E[2][2]/IOTA))*cstate[2] + (-(r_dot*d[2][1]))*cstate[1]
    return c_dot
#--------------------------------------------------------------------------------------------------#

def rk4(r_dot, cstate, V, E, t_step):
    
    c_dot = equation(r_dot, cstate, V, E)
    
    cstate1=cstate[1]
    cstate2=cstate[2]

    x1=complex(t_step*c_dot[1])
    y1=complex(t_step*c_dot[2])

    cstate[1]=cstate1+x1/2.0e0
    cstate[2]=cstate2+y1/2.0e0
    
    c_dot = equation(r_dot, cstate, V, E)

    x2=complex(t_step*c_dot[1])
    y2=complex(t_step*c_dot[2])

    cstate[1]=cstate1+x2/2.0e0
    cstate[2]=cstate2+y2/2.0e0
    
    x2=x1+2.0e0*x2
    y2=y1+2.0e0*y2

    c_dot = equation(r_dot, cstate, V, E)

    x3=complex(t_step*c_dot[1])
    y3=complex(t_step*c_dot[2])

    cstate[1]=cstate1+x3
    cstate[2]=cstate2+y3
    
    x3=x2+2.0e0*x3
    y3=y2+2.0e0*y3

    c_dot = equation(r_dot, cstate, V, E)

    x4=(x3+complex(t_step*c_dot[1]))/6.0e0
    y4=(y3+complex(t_step*c_dot[2]))/6.0e0

    cstate[1]=cstate1+x4
    cstate[2]=cstate2+y4
    return cstate
#--------------------------------------------------------------------------------------------------#

def hopping_probability(r_dot,cstate,V, nstate, mass, E, t_step,x):

    #Coupling : Non-Adiabatic coupling vector 'd_ij'
    z = (V[1][1] - V[2][2])/(V[1][2]*2e0)
    d[1][2] = 1e0/(2e0*(1+z**2)) * ( ( 1e0/(2e0*V[1][2]**2)) * ( V[1][2]*(der_V[1][1]-der_V[2][2]) - der_V[1][2]*(V[1][1]-V[2][2]) ) )
    d[2][1] = -d[1][2]
    d[1][1] = 0
    d[2][2] = 0

    a[1][2]=cstate[1]*cstate[2].conjugate()
    a[2][1]=cstate[2]*cstate[1].conjugate()
    a[1][1]=cstate[1]*cstate[1].conjugate()
    a[2][2]=cstate[2]*cstate[2].conjugate()
    
    b[1][2]=2e0*( a[1][2].conjugate()*E[1][2] ).imag - 2e0*( (a[1][2]).conjugate()*r_dot*d[1][2] ).real
    b[2][1]=2e0*( a[2][1].conjugate()*E[2][1] ).imag - 2e0*( (a[2][1]).conjugate()*r_dot*d[2][1] ).real

    rnd = random.random()
    hop = 0

    if (nstate==1) & ( (0.5e0*mass*(r_dot)**2) > (E[2][2]-E[1][1]) ):
        prob_hop = (t_step*b[2][1])/(a[1][1]).real
        if (prob_hop>rnd):
            newstate=2
            hop=1              
    elif (nstate==2):
        prob_hop = (t_step*b[1][2])/(a[2][2]).real
        if (prob_hop>rnd): 
            newstate=1
            hop=1                                      
        
    if (hop==1):
        r_dot  =  (r_dot/abs(r_dot))*math.sqrt(2.0e0*(E[nstate][nstate]-E[newstate][newstate]+0.5*mass*r_dot**2)/mass)
        nstate = newstate

    return nstate, r_dot
#--------------------------------------------------------------------------------------------------#

def velocity_verlet(x, r_dot, mass,F, t_step, nstate ):

    i=nstate
    acc   = F[i]/mass
    x     = x     + r_dot*t_step + 0.5*acc*(t_step**2)

    V, der_V  = potential(x)
    F         = force(V, der_V)
    E         = eigenvalues(V)

    acc       = 0.5*(acc + F[i]/mass )
    r_dot     = r_dot + acc*t_step
    
    return x, r_dot, V, F, E
#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
	start_time = time.time()
	main_code()



