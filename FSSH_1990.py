# Surface Hopping - Tully(1990)
# 
import math, cmath
import random



V      = [[0,0,0], [0,0,0], [0,0,0]]     #[[]]
der_V  = [[0,0,0], [0,0,0], [0,0,0]] 
E      = [[0,0,0], [0,0,0], [0,0,0]] 
F      = [[0],     [0],     [0]    ] 
a      = [[0,0,0], [0,0,0], [0,0,0]]
b      = [[0,0,0], [0,0,0], [0,0,0]]  
d      = [[0,0,0], [0,0,0], [0,0,0]] 
c_dot  = [[0,0,0], [0,0,0], [0,0,0]] 


#IOTA   = complex(0,1)

###################################################################################################
def main_code():
    
    # setup()
    #global ntraj, t_step, nstate, newstate, mass, momentum, x

    ntraj     = 2000
    t_step    = 1       # dt_step

    #nstate    = 1
    #cstate    = [0,complex(1,0),complex(0,0)]
    mass      = 2.0e3

    #x         = -9.9999
    xmin,xmax = -10.0,10.0

    for momentum in range(0,31):

        #r_dot = momentum/mass
        j=0
        k=0
 
        for nt in range(0,ntraj):
            x         = -9.9999
            r_dot = momentum/mass
            nstate    = 1
            cstate    = [0,complex(1,0),complex(0,0)]
            t_evolve = 0
            while(x>xmin and x<xmax):
                V, der_V  = potential(x)
                F         = force(V, der_V)
                E         = eigenvalues(V)
            
                t_evolve = t_evolve + t_step 

                cstate    = rk4(r_dot, cstate, V, E, t_step)
                nstate    = hopping_probability(r_dot,cstate,V, nstate, mass, E, t_step, x)
                #print(nstate)
                x, r_dot  = velocity_verlet(x, r_dot, F, mass, t_step, nstate )
            
            if (nstate==1):
                j=j+1
                #print("inside -j")
            else:
                k=k+1  
                #print("inside -k")
        prob1_trans=float(j/ntraj)
        prob2_trans=float(k/ntraj)

        print("--->",momentum,j,k,"**",prob1_trans,prob2_trans, "-->", t_evolve)
#--------------------------------------------------------------------------------------------------#

'''
def setup():
    # read input file 
    ntraj  = 200
    t_step = 1
    nstate  = 1
    newstate= 2
    mass    = 2e3
    #n       #gridpoints to plot V etc
#--------------------------------------------------------------------------------------------------#

def initialize():
    global cstate
    global r_dot
    #x = -9.9999
    r_dot = momentum/mass
    cstate = [0,complex(1,0),complex(0,0)]
    #nstate = 1
    # V      = [[0,0], [0,0]]
    # der_V  = [[0,0], [0,0]]
#--------------------------------------------------------------------------------------------------#
'''

def potential(x):

    #Potential and it's derivatives
    A,B,C,D = 0.01e0, 1.6e0, 0.005e0, 1e0

    if (x>0): 
        V[1][1]      =  A*(1-math.exp(-B))
        der_V[1][1]  =  A*B*(math.exp(-B))
    elif (x<0): 
        V[1][1]      = -A*(1-math.exp(B))
        der_V[1][1]  =  A*B*(math.exp(B))
    
    V[2][2]          = -V[1][1]
    der_V[2][2]      = -der_V[1][1]
    V[1][2]          = C*(math.exp(-D*x**2))
    der_V[1][2]      = -2*C*D*(math.exp(-D*x**2))
    V[2][1]          = V[1][2]
    der_V[2][1]      = der_V[1][2]
    return V, der_V
#--------------------------------------------------------------------------------------------------#

def force(V, der_V):
    
    F[1]=(1/math.sqrt((V[1][1]**2)+(V[1][2]**2)))*(V[1][1]*der_V[1][1]) +(V[1][2]*der_V[1][2])
    F[2]=-(1/math.sqrt((V[1][1]**2)+(V[1][2]**2)))*(V[1][1]*der_V[1][1]) +(V[1][2]*der_V[1][2])
    return F
#--------------------------------------------------------------------------------------------------#

def eigenvalues(V):

    E[2][2] = (V[1][1]+V[2][2])+math.sqrt((V[1][1]+V[2][2])**2-(4*(V[1][1]*V[2][2]-(V[1][2])**2)))/2.e0
    E[1][1] = (V[1][1]+V[2][2])-math.sqrt((V[1][1]+V[2][2])**2-(4*(V[1][1]*V[2][2] -(V[1][2])**2)))/2.e0
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
    #newstate  = 2
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
    #print(nstate,x,"##", a[2][1], b[2][1],"##",d[1][2],"##", r_dot, rnd)

    if (nstate==1):
        if ((0.5e0*mass*(r_dot)**2)>(E[2][2]-E[1][1])):
            prob_hop = (t_step*b[2][1])/(a[1][1]).real
            #print(nstate,x,"##", r_dot, rnd, prob_hop)
            if (prob_hop>rnd):
                newstate=2
                hop=1
                r_dot=math.sqrt(2.0e0*(E[1][1]-E[2][2]+(0.5*mass*r_dot**2))/mass)
            else: 
                hop=0
                nstate=nstate      
        else:
            nstate=nstate       
    elif (nstate==2):
        prob_hop = (t_step*b[1][2])/(a[2][2]).real
        #print(nstate,x,"##", r_dot, rnd, prob_hop)
        if (prob_hop>rnd): 
            newstate=1
            hop=1
            if (r_dot>0):
                r_dot =  math.sqrt(2.0e0*(E[2][2]-E[1][1]+0.5*mass*r_dot**2)/mass)
            else: 
                r_dot = -math.sqrt(2.0e0*(E[2][2]-E[1][1]+0.5*mass*r_dot**2)/mass)       
        else: 
            hop=0
            nstate=nstate
        
    if (hop==1):
        nstate=newstate

    return nstate
#--------------------------------------------------------------------------------------------------#

def velocity_verlet(x, r_dot, F, mass, t_step, nstate ):

    i=nstate
    acc   = F[i]/mass
    x     = x     + r_dot*t_step + 0.5*acc*(t_step**2)
    V, der_V  = potential(x)
    F         = force(V, der_V)
    acc       = 0.5*(acc + F[i]/mass )
    r_dot = r_dot + acc*t_step
    #print energy
    return x, r_dot

#--------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    main_code()



