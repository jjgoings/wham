import numpy as np
import re
import sys

class WHAM(object):
    def __init__(self,T,metadata):
        self.metadata = metadata
        self.kbolt    = 0.001982923700 # Boltzmann's constant in kcal/mol K
        self.T        = T
        self.B        = 1.0/(self.kbolt*self.T)
        self.files    = []
        self.restraints = []
        self.forces = []
        self.restraint_coordinate = []
        self.computed_free = False  # flag for plotting PMF
        # get windows and parameters
        with open(self.metadata) as fp:
            for line in fp:
                if not line.strip() or (line[0] == '#'):
                    # ignore blank lines or comments
                    continue
                else:
                    data = line.split()
                    self.files.append(data[0])
                    self.restraints.append(float(data[1]))
                    self.forces.append(float(data[2]))
                    try:
                        self.restraint_coordinate.append(int(data[3]))
                    except IndexError:
                        sys.exit('Need to add reaction coordinate to metadata')
        # make sure dimension of data is consistent
        assert(len(self.files) == len(self.restraints) 
               == len(self.forces) == len(self.restraint_coordinate))

        # add trajectory reaction coordinate data 
        #FIXME assumes all trajectory same length!
        for idx,window in enumerate(self.files):
            # restraint coordinate ensures trajectory data is consistent with
            # the applied restraint (e.g. restraint and traj data in same basis)
            R     = np.loadtxt(window,usecols=[self.restraint_coordinate[idx]])
            if idx == 0:
                self.R     = R
            else:
                self.R     = np.vstack((self.R,R))

        self.Nwind = self.R.shape[0]
        self.nt    = self.R.shape[1]

    def compute_free(self,maxiter=10000,conver=1e-6):
        ebw = np.zeros((self.Nwind,self.nt,self.Nwind))
        ebf = np.ones((self.Nwind))
        fact = self.nt*ebf
        ebf2 = np.zeros_like(fact)

        # precompute exp(-BW_k(R_i,k))
        for k in range(self.Nwind):
            for i in range(self.Nwind):
                ebw[i,:,k] += np.exp(-self.B*0.5*self.forces[k]*
                                      (self.R[i] - self.restraints[k])**2)

        oldebf = np.zeros_like(ebf)
        for n in range(maxiter):
            for k in range(self.Nwind):

                denom = 1.0/np.einsum('ilj,j->il',ebw,fact)
                ebfk = np.einsum('il,il',ebw[:,:,k],denom)
        
                ebf2[k] = ebfk
                ebf[k] = 1.0/(ebf[0]*ebfk)
                fact[k] = self.nt*ebf[k]
        
            delta = np.linalg.norm(np.log(ebf*ebf2))
            ebf = ebf2[0]/ebf2
            self.f = np.log(ebf)/self.B
            if delta < conver:
                print("Converged free-energies: \n",self.f)
                np.savetxt('free-energies.txt',self.f,fmt='%.8f',delimiter=',')
                self.computed_free = True
                break
            else:
                if n % 50 == 0:
                    print("Free energies at iteration "+str(n)+": \n",self.f)


    def compute_pmf(self,hmin,hmax,num_bins,pmf_crd=1,plot=True):

        self.hmin = hmin
        self.hmax = hmax
        self.num_bins = num_bins
        self.pmf_coordinate = pmf_crd
        self.do_plot = plot

        if not self.computed_free:
            self.f = np.loadtxt('./free-energies.txt')

        # Load desired reaction coordinate for PMF
        for idx,window in enumerate(self.files):
            R_pmf = np.loadtxt(window,usecols=[self.pmf_coordinate])
            if idx == 0:
                self.R_pmf = R_pmf 
            else:
                self.R_pmf = np.hstack((self.R_pmf,R_pmf))
        
        # generate weights
        R = self.R.reshape(self.nt*self.Nwind,)
        weights = 0.0
        for i in range(self.Nwind):
            weights += np.exp(-self.B*(0.5*self.forces[i]
                       *(R - self.restraints[i])**2 - self.f[i]))

        weights = 1.0/weights
        
        bin_width = (self.hmax - self.hmin)/self.num_bins
       
        # FIXME better way to do with numpy linspace? 
        bins = np.arange(hmin,hmax+0.000001,bin_width)
        
        self.pdf,b = np.histogram(self.R_pmf,bins=bins,weights=weights,
                                  density=True)

        # self.bins is now the center of each bin        
        self.bins = bins[:-1] + 0.5*np.diff(bins)

        self.pmf = -(1./self.B)*np.log(self.pdf) 

        if self.do_plot:
            import matplotlib.pyplot as plt
            plt.plot(self.bins,self.pmf-min(self.pmf))
            plt.show()

if __name__ == "__main__":
    wham = WHAM(300.0,'metadata')
    wham.compute_free()
    wham.compute_pmf(-2.0,2.0,40)

