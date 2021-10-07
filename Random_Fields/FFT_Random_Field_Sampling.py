import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import beta
from tqdm import tqdm

def PSDF_2D(w,k, ell1,ell2, sig):
    """ Calculate the Power Spectral Density, see eq. (65)."""
    praefac = sig*(ell1)*(ell2)/ (4*pi)
    xi1 =  np.sum(w**2,1).reshape(-1,1) + np.sum(w**2,1) 
    xi2 =  np.sum(k**2,1).reshape(-1,1) + np.sum(k**2,1)  
    ker = praefac * np.exp(-.25* (xi1 * ell1**2 + xi2 * ell2**2))
    return(ker / np.sum(ker))

def Lognstat(mu, sigma):
    """Calculate the mean of and variance of the Lognormal distribution given
    the mean (`mu`) and standard deviation (`sigma`), of the associated normal 
    distribution."""
    m = np.exp(mu + sigma**2 / 2.0)
    var = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)
    return (m, var)

def Gamma(sample_field,n):
    """Calculate a Gamma field as in eq. (95)"""
    lim = int(2*n)
    G = np.sum([ (np.square(np.array(sample_field)[i])) for i in range(lim)],0)
    return(G)

def Beta(Gamma1, Gamma2):
    """Calculate a Beta field as in eq. (100)"""
    return(Gamma1 / (Gamma1 + Gamma2))


def Logn(sample_field,mu_g, sig_g):
    """Calculate a Lognormal field as in eq. (108)"""
    log_field = np.exp(mu_g + sig_g*sample_field)
    log_mu,log_sig = Lognstat(mu_g,sig_g)
    return(log_field,  log_mu, log_sig)

def Unif(Gamma1, Gamma2):
    """Calculate a Uniformly disrtibuted field as a special case of Beta field"""
    return(0.5 * Gamma1 / (0.5* Gamma1 + 0.5* Gamma2))

def mult(A,B):
    return(A*B)  

# Frequency domain
Nw = 2**7
Nk = 2**7

dw = 0.0781
dk = 0.0781

kmax = dk*Nk
wmax = dw*Nw

w = np.dot(dw , range(0,Nw-1)).reshape(-1,1)
k = np.dot(dk ,range(0,Nk-1)).reshape(-1,1)

# Spatial Domain
Mw = 2*Nw
Mk = 2*Nk

t = 2*pi/dw *np.linspace(0,1,Mw+1)
Mt = len(t)
x = 2*pi/dk *np.linspace(0,1,Mk+1)
Mx = len(x)

# Define field correlation lengthscales
corr1 = 10
corr2 = 10
sig = 1

S_freq_domain = PSDF_2D(w,k,corr1,corr2,sig)
S_spatial_domain = np.zeros(shape =(Mk,Mw))
S_spatial_domain[0:Nk-1,0:Nw-1] = S_freq_domain 


Log_fields = []
Beta_fields = []
Uni_fields = []
Gaussian_fields =  []

Beta_22  = []
Beta_0505= []
Beta_24= []
Beta_41= []
  
set_seed = 12
for numbers in tqdm(range(0,10)):

    gamma_8 = []
    gamma_4 = []

    for m in (range(set_seed)):
        np.random.seed(m+numbers*set_seed)
       
        phi1 = np.random.rand(Mk,Mw)*2*pi
        phi2 = np.random.rand(Mk,Mw)*2*pi
        B1 = 2*np.array(list(map(mult,np.sqrt(S_spatial_domain*dk*dw), np.exp(1j*phi1))))
        B2 = 2*np.array(list(map(mult,np.sqrt(S_spatial_domain*dk*dw), np.exp(1j*phi2))))
     
        F1 =  (Mk*2*pi) * np.fft.ifft(B1,Mx,0)
        F2 = (Mk*2*pi)  * np.fft.ifft(B2,Mx,0)
        
        F1 = Mw * np.fft.ifft(F1,Mt,1)
        F2 =   np.fft.fft(F2,Mt,1)
        
        
        # Sample of Gaussian Random Field via FFT-method
        y = np.real(F1+F2)
        Gaussian_fields.append(y) 
        
        
        if m < 8:
            gamma_8.append(y)
        if m > 7 :
            gamma_4.append(y)


        
    Gamma4 = np.array(Gamma(gamma_8,4))
    Gamma2 = np.array(Gamma(gamma_4,2)) 
    Uniform1 = np.array(Gamma(gamma_8,1))
    Uniform2 = np.array(Gamma(gamma_4,1))
    Gamma1_1 = np.array(Gamma(gamma_8,1))
    Gamma1_2 = np.array(Gamma(gamma_8,2))
    Gamma2_2 = np.array(Gamma(gamma_8,2)) 
    Gamma05_1 = np.array(Gamma(gamma_8,.5))
    Gamma05_2 = np.array(Gamma(gamma_4,.5))
    
    Beta_22.append(Beta(Gamma2,Gamma2_2))
    Beta_0505.append(Beta(Gamma05_1,Gamma05_2))
    Beta_24.append(Beta(Gamma2 ,Gamma4))
    Beta_41.append(Beta(Gamma4,Gamma1_1))
    
    
    Log_fields.append(Logn(y,np.mean(y),np.std(y)))
    Beta_fields.append(Beta(Gamma4,Gamma2))
    Uni_fields.append(Unif(Uniform1, Uniform2))

# Plot the result



def PlotBetaPdf():
    """  Define the distribution parameters to be plotted """
    alpha_values = [4, 2,2,  4]
    beta_values =  [2, 2,4,  1]
    linestyles = ['-', '--', ':','-']
    x = np.linspace(0, 1, 1002)[1:-1]

    for a, b, ls in zip(alpha_values, beta_values, linestyles):
        dist = beta(a, b)
    
        plt.plot(x, dist.pdf(x), ls=ls,
                 label=r'$m=%.1f,\ n=%.1f$' % (a, b), color= 'k')
    
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    
    plt.xlabel('$x$', size = 22)
    plt.ylabel(r'$p(x|m,n)$', size = 22)
    leg = plt.legend(loc=0, fontsize = 18)
    plt.tick_params(labelsize=20)
    plt.show()
    plt.tight_layout()
    return
    
i = 0 # index for plot, number in [0,10]
fig = plt.figure(figsize=(5, 7))
plt.subplots_adjust(wspace= 0.25, hspace= 0.25)

sub1 = fig.add_subplot(3,2,1) 
ax = plt.gca()
plt.axis('off')
plt.title('Beta(4,2)', size = 16)
im = ax.imshow(Beta_fields[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=16) 

sub2 = fig.add_subplot(3,2,2) # two rows, two columns, second cell
ax = plt.gca()
plt.axis('off')
plt.title('Beta(2,2)', size = 16)
im = ax.imshow(Beta_22[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=16) 

sub3 = fig.add_subplot(3,2,3)
ax = plt.gca()
plt.axis('off')
plt.title('Beta(2,4)', size = 16)
im = ax.imshow(Beta_24[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=16) 

sub4 = fig.add_subplot(3,2,4)
ax = plt.gca()
plt.axis('off')
plt.tick_params(labelsize=16)
plt.title('Beta(4,1)', size = 16)
im = ax.imshow( Beta_41[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=16) 
sub5 = fig.add_subplot(3,2,(5,6)) 

plt.tight_layout()
PlotBetaPdf()
