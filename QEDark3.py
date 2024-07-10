
# # from QEDarkConstants import *
# #attempting to implement QEDark with Pytorch tensors instead

        

class DM_Halo_Distributions:
    def __init__(self,V0=None,VEarth=None,VEscape=None,RHOX=None,crosssection=None):
        #inputs in km/s
        from QEDarkConstants import nq,dQ,Emin,Emax,dE,lightSpeed_kmpers

    
        self.nq = nq
        self.dQ = dQ
        self.Emin = Emin
        self.Emax = Emax
        self.dE = dE
        self.device='cpu'

        if V0 is None:
            from QEDarkConstants import v0
            self.v0 = v0/lightSpeed_kmpers
        else:
            self.v0 = V0/lightSpeed_kmpers


        if VEarth is None:
            from QEDarkConstants import vEarth
            self.vEarth = vEarth/lightSpeed_kmpers
        else:
            self.vEarth = VEarth/lightSpeed_kmpers
        
        if VEscape is None:
            from QEDarkConstants import vEscape
            self.vEscape = vEscape/lightSpeed_kmpers
        else:
            self.vEscape = VEscape/lightSpeed_kmpers

        if RHOX is None:
            from QEDarkConstants import rhoX
            self.rhoX = rhoX
        else:
            self.rhoX = RHOX
        if crosssection is None:
            from QEDarkConstants import crosssection
            self.cross_section = crosssection

        from numpy import concatenate,arange,array
        self.default_masses = concatenate((arange(0.2,0.8,0.025),array([0.9]),arange(1,5,0.05),arange(5,11,1),array([20,50,100,200,500,1000,10000])))*1e6 # in eV


    def optimize(self, device):
        self.device = device



    def generate_halo_files(self,mX,model):
        import numpy as np
        #takes in mass [eV]
        #model is a string key
        #params depends on which halo function you are calling
        vMins =[]
        for Ee in np.round(np.arange(self.Emin,self.Emax,self.dE),2): 
            for qi in range(1,self.nq):
                vMin = self.vmin(Ee,qi,mX)
                if vMin < (self.vEscape+self.vEarth)*2: #naive cutoff
                    vMins.append(vMin)
        vMins = np.array(vMins)
        vMins  = np.sort(vMins)
        #this next step could use some vectorization, but I will be a bit lazy here
        etas = []
        for v in vMins:
            if model == 'shm':
                eta = self.etaSHM(v)
                #eta = etaSHM(vmin,params) # (cm/s)^-1 
            elif model == 'tsa':
                eta = self.etaTsa(v)
            elif model == 'dpl':
                eta = self.etaDPL(v)
            # elif model == 'msw':
            #     eta = etaMSW(v,_params)
            # elif model == 'debris':
            #     eta = etaDebris(v,_params)
            else:
                print("Undefined halo parameter. Options are ['shm','tsa','dpl','msw','debris']")
            etas.append(eta)
        etas = np.array(etas)
        mass_string = mX*1e-6 #turn into MeV
        # print(mass_string)
        mass_string = np.round(mass_string,3)
        if mass_string.is_integer():
            mass_string = int(mass_string)
        else:
            mass_string = str(mass_string)
            mass_string = mass_string.replace('.',"_")

        # print(mass_string)
        with open(f'./halo_data/{model}/mDM_{mass_string}_MeV.txt','w') as f:
            for i in range(len(vMins)):
                f.write(f"{vMins[i]}\t{etas[i]}\n") #[km/s], [s/km]
        return
    

    def vmin(self,EE,qin,mX):
        from  QEDarkConstants import lightSpeed,dQ
        # q = qin * alpha *me_eV #this was originally in the code, but the second line was in QEDark. using that instead
        q= qin*dQ
        return ((EE/q)+(q/(2*mX)))


    def vmin_tensor(self,Ee_array,q_array,mX):
        from  QEDarkConstants import lightSpeed,dQ
        import torch
        # q = qin * alpha *me_eV #this was originally in the code, but the second line was in QEDark. using that instead
        q= q_array*dQ
        q_tiled = torch.tile(q,(len(Ee_array),1))
        EE_tiled = torch.tile(Ee_array,(len(q),1)).T

        return ((EE_tiled/q_tiled)+(q_tiled/(2*mX))) # units of c

    # def eta_MB(self,vMin):    #same as SHM but the indefinite integration is done so this is faster. In units of inverse vMin
    #     import numpy as np
    #     from scipy.special import erf
    #     if (vMin < self.vEscape - self.vEarth):
    #         val = -4.0*self.vEarth*np.exp(-(self.vEscape/self.v0)**2) + np.sqrt(np.pi)*self.v0*(erf((vMin+self.vEarth)/self.v0) - erf((vMin - self.vEarth)/self.v0))
    #     elif (vMin < self.vEscape + self.vEarth):
    #         val = -2.0*(self.vEarth+self.vEscape-vMin)*np.exp(-(self.vEscape/self.v0)**2) + np.sqrt(np.pi)*self.v0*(erf(self.vEscape/self.v0) - erf((vMin - self.vEarth)/self.v0))
    #     else:
    #         val = 0.0
    #     K = (self.v0**3)*(-2.0*np.pi*(self.vEscape/self.v0)*np.exp(-(self.vEscape/self.v0)**2) + (np.pi**1.5)*erf(self.vEscape/self.v0))
    #     return (self.v0**2)*np.pi/(2.0*self.vEarth*K)*val
    
    def etaSHM(self,vMin):
        from scipy.integrate import quad, dblquad, nquad
        from scipy.special import erf
        from QEDarkConstants import lightSpeed_kmpers,lightSpeed
        import numpy as np
        """
        Standard Halo Model with sharp cutoff. 
        Fiducial values are v0=220 km/s, vE=232 km/s, vesc= 544 km/s
        params = [v0, vE, vesc]
        """
        v0 = self.v0
        vE = self.vEarth
        vesc = self.vEscape
        KK=v0**3*(-2.0*np.exp(-vesc**2/v0**2)*np.pi*vesc/v0+np.pi**1.5*erf(vesc/v0))
    #    print('KK=',KK)
        def func(vx2):
            return np.exp(-vx2/(v0**2))

        if vMin <= vesc - vE:
            # eq. B4 from 1509.01598
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/KK)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0] * (1/lightSpeed) * 1e-2
        elif vesc - vE < vMin <= vesc + vE:
            # eq. B5 from 1509.01598
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)] 
            def bounds_vX():
                return [vMin, vE+vesc]
            def eta(cosq,vx):
                return (2*np.pi/KK)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0] * (1/lightSpeed) * 1e-2
        else:
            return 0

    def eta_MB_tensor(self,vMin_tensor):
        """Integrated Maxwell-Boltzmann Distribution"""

        import torch
        from QEDarkConstants import lightSpeed
        eta = torch.zeros_like(vMin_tensor,device=self.device)


        val_below = -4.0*self.vEarth*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + torch.sqrt(torch.tensor(torch.pi,device=self.device))*self.v0*(torch.erf((vMin_tensor+self.vEarth)/self.v0) - torch.erf((vMin_tensor - self.vEarth)/self.v0))

        val_above = -2.0*(self.vEarth+self.vEscape-vMin_tensor)*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + torch.sqrt(torch.tensor(torch.pi,device=self.device))*self.v0*(torch.erf(torch.tensor(self.vEscape/self.v0,device=self.device)) - torch.erf((vMin_tensor - self.vEarth)/self.v0))


        eta = torch.where(vMin_tensor < self.vEscape + self.vEarth, val_above,eta)
        eta = torch.where(vMin_tensor < self.vEscape - self.vEarth, val_below,eta)

        K = (self.v0**3)*(-2.0*torch.pi*(self.vEscape/self.v0)*torch.exp(torch.tensor(-(self.vEscape/self.v0)**2,device=self.device)) + (torch.pi**1.5)*torch.erf(torch.tensor(self.vEscape/self.v0,device=self.device)))

        etas = (self.v0**2)*torch.pi/(2.0*self.vEarth*K)*eta #units of c^-1
        etas/=lightSpeed #convert to s/m
        etas*=1e-2 #convert to s/cm
        #not sure if etas is allowed to be zero.
        etas = torch.where(etas < 0,0,etas)

        return etas
    

    def etaTsa(self,vMin):

        from scipy.integrate import nquad
        from QEDarkConstants import q_Tsallis,lightSpeed
        import numpy as np
        """
        Tsallis Model, q = .773, v0 = 267.2 km/s, and vesc = 560.8 km/s
        give best fits from arXiv:0902.0009. 
        params = [v0, vE, q]
        """
        q = q_Tsallis

        if q <1:
            vesc = self.v0/np.sqrt(1-q)
        else:
            vesc = self.vEscape # this is old, updated to new parameter 560.8e5 # cm/s
    #    vesc = 544*kmpers_nu ## to test against SHM    
        def func(vx2):
            if q == 1:
                return np.exp(-vx2/self.v0**2)
            else:
                return (1-(1-q)*vx2/self.v0**2)**(1/(1-q))
        " calculate normalization constant "
        def inttest(vx):
            if q == 1:
                if vx <= vesc:
                    return vx**2*np.exp(-vx**2/self.v0**2)
                else:
                    return 0 
            else:
                if vx <= vesc:
                    return vx**2*(1-(1-q)*vx**2/self.v0**2)**(1/(1-q))
                else:
                    return 0            
        def bounds():
            return [0.,vesc]
        K_=4*np.pi*nquad(inttest,[bounds])[0]

    #    K_ = 4/3*np.pi*vesc**3*hyp2f1(3/2, 1/(q-1), 5/2, (1-q)*vesc**2/v0**2) # analytic expression, runs faster
        
        if vMin <= vesc - self.vEarth:
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*self.vEarth+np.sqrt((cosq**2-1)*self.vEarth**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/K_)*vx*func(vx**2+self.vEarth**2+2*vx*self.vEarth*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0]* (1/lightSpeed) * 1e-2
            
        elif vesc - self.vEarth < vMin <= vesc + self.vEarth:
            def bounds_cosq(vx):
                return [-1, (vesc**2-self.vEarth**2-vx**2)/(2*vx*self.vEarth)] 
            def bounds_vX():
                return [vMin, self.vEarth+vesc]
            def eta(cosq,vx):
                return (2*np.pi/K_)*vx*func(vx**2+self.vEarth**2+2*vx*self.vEarth*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0]* (1/lightSpeed) * 1e-2
        else:
            return 0   


    def etaDPL(self,vMin):
        from QEDarkConstants import k_DPL,lightSpeed
        from scipy.integrate import nquad
        import numpy as np
        """
        Double Power Law Profile, 1.5 <= k <= 3.5 found to give best fit to N-body
        simulations. 
        takes input velocities in km/s
        params = [vMin, v0, vE, vesc, k]
        """
        v0 = self.v0
        vE = self.vEarth
        vesc = self.vEscape
        k = k_DPL
        
        def func(vx2):
            return (np.exp((vesc**2-vx2)/(k*v0**2))-1)**k
        " calculate normalization constant "
        def inttest(vx):
            if vx <= vesc:
                return vx**2*(np.exp((vesc**2-vx**2)/(k*v0**2))-1)**k
            else:
                return 0
        def bounds():
            return [0.,vesc]
        K_=4*np.pi*nquad(inttest,[bounds])[0]   
        
        if vMin <= vesc - vE:
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [vMin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
            def eta(vx,cosq):
                return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_vX,bounds_cosq])[0]* (1/lightSpeed) * 1e-2
            
        elif vesc - vE < vMin <= vesc + vE:
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)] 
            def bounds_vX():
                return [vMin, vE+vesc]
            def eta(cosq,vx):
                return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(eta, [bounds_cosq,bounds_vX])[0]* (1/lightSpeed) * 1e-2
        else:
            return 0

    def step_function_eta(self,vMins, params):
        #vMins is 2d array
        import torch
        from QEDarkConstants import lightSpeed,lightSpeed_kmpers
        num_steps = params.shape[0]
        # vMax = (vEarth + vEscape)*1.1 #can change this later
        vMax = 1000 #km /s
        vMax/=lightSpeed_kmpers #units of c
        
        vis = torch.arange(0,vMax,step = vMax/num_steps,device=self.device)
        if params.device == 'mps':
            params
        step_heights = torch.exp(params)


        vMins_tiled = torch.tile(vMins[:,:,None],(1,1,num_steps))
        vMins_tiled = vMins_tiled.permute(*torch.arange(vMins_tiled.ndim - 1, -1, -1,device=self.device))
        vis_tiled = torch.tile(vis[:,None,None],(1,vMins.shape[1],vMins.shape[0]))
        step_heights_tiled = torch.tile(step_heights[:,None,None],(1,vMins.shape[1],vMins.shape[0]))

        temp = vis_tiled-vMins_tiled
        heaviside = torch.where(temp > 0, 1,0)

        etas =  torch.sum(step_heights_tiled * heaviside,axis=0).T
        # #normalize
        etas*=1e-15

        return etas #same shape as vMins

        




class QEDark:
    def __init__(self):
        from QEDarkConstants import v0,vEarth,vEscape,rhoX,crosssection,nq,dQ,Emin,Emax,dE
        self.v0 = v0
        self.vEarth = vEarth
        self.vEscape = vEscape
        self.rhoX = rhoX
        self.cross_section = crosssection
        self.ionization_func = self.p100_function
        self.nq = nq
        self.dQ = dQ
        self.Emin = Emin
        self.Emax = Emax
        self.dE = dE
        self.device = 'cpu'

        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX,self.cross_section)

    def optimize(self,device):
        self.device = device
        self.DM_Halo.optimize(device)

    def update_params(self,v0,vEarth,vEscape,rhoX,crosssection,model=None,masses=None):
        #masses must be in eV if passed in 
        self.v0 = v0
        self.vEarth = vEarth
        self.vEscape = vEscape
        self.rhoX = rhoX
        self.cross_section = crosssection
        self.DM_Halo = DM_Halo_Distributions(self.v0,self.vEarth,self.vEscape,self.rhoX)
        
        if model is not None:
            if masses is not None:
                loop_masses = masses
            else:
                loop_masses = self.DM_Halo.default_masses
            for mX in loop_masses:
                self.DM_Halo.generate_halo_files(mX,model)
    
    def update_crosssection(self,crosssection):
        self.cross_section = crosssection

    def update_Emin(self,val):
        self.Emin = val


    def change_to_step(self):
        self.ionization_func = self.step_function

    def getClosest(self,val1, val2, target):
    
        if (target - val1 >= val2 - target):
            return val2
        else:
            return val1
        
    def findClosest(self,arr, n, target):
        # Corner cases
        if (target <= arr[0]):
            return arr[0]
        if (target >= arr[n - 1]):
            return arr[n - 1]
    
        # Doing binary search
        i = 0; j = n; mid = 0
        while (i < j): 
            mid = (i + j) // 2
    
            if (arr[mid] == target):
                return arr[mid]
    
            # If target is less than array 
            # element, then search in left
            if (target < arr[mid]) :
    
                # If target is greater than previous
                # to mid, return closest of two
                if (mid > 0 and target > arr[mid - 1]):
                    return self.getClosest(arr[mid - 1], arr[mid], target)
    
                # Repeat for left half 
                j = mid
            
            # If target is greater than mid
            else :
                if (mid < n - 1 and target < arr[mid + 1]):
                    return self.getClosest(arr[mid], arr[mid + 1], target)
                    
                # update i
                i = mid + 1
            
        # Only single element left after search
        return arr[mid]




    def FDM(self,q_eV,n):
        from QEDarkConstants import alpha,me_eV
        """
        DM form factor
        n = 0: FDM=1, heavy mediator
        n = 1: FDM~1/q, electric dipole
        n = 2: FDM~1/q^2, light mediator
        """
        return (alpha*me_eV/q_eV)**n

    def mu_Xe(self,mX):
        from QEDarkConstants import me_eV
        """
        DM-electron reduced mass
        """
        return mX*me_eV/(mX+me_eV)

    def mu_XP(self,mX):
        from QEDarkConstants import mP_eV
        """
        DM-proton reduced mass
        """
        return mX*mP_eV/(mX+mP_eV)

    def TFscreening(self,q_arr, E_array, DoScreen):
        from QEDarkConstants import eps0,alphaS,qTF,me_eV,omegaP
        import torch
        q_arr_tiled= torch.tile(q_arr,(len(E_array),1))
        if DoScreen==True:
            E_array_tiled= torch.tile(E_array,(len(q_arr),1)).T
            val = 1.0/(eps0 - 1) + alphaS*((q_arr_tiled/qTF)**2) + q_arr**4/(4.*(me_eV**2)*(omegaP**2)) - (E_array_tiled/omegaP)**2
            return 1./(1. + 1./val)
        else:
            return torch.ones_like(q_arr_tiled)

    def get_halo_data(self,mX,q_tensor,Ee_tensor,FDMn,halo_model,isoangle=None,halo_id_params=None,forceCalculate=False,useVerne=False,calcErrors=None):
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator    
        from QEDarkConstants import lightSpeed,ccms,sec2yr  
        import torch
        import os
        import re

        
        mass_string = mX*1e-6 #turn into MeV

        mass_string = float(mass_string)
        from numpy import round as npround
        mass_string = npround(mass_string,3)
        if isoangle is None:
            if mass_string.is_integer():
                mass_string = int(mass_string)
            else:
                mass_string = str(mass_string)
                mass_string = mass_string.replace('.',"_")

        else:
            mass_string = str(mass_string)
            mass_string = mass_string.replace('.',"_")
        sigmaE_str = str(self.cross_section)
        sigmaE_str.replace('.',"_")
        discrepancy = 1
        # file_etas = []
        # file_vmins = []
        if halo_id_params is not None:
            vMins = self.DM_Halo.vmin_tensor(Ee_tensor,q_tensor,mX)
            # print(vMins.shape,halo_id_params.shape)
            etas = self.DM_Halo.step_function_eta(vMins, halo_id_params) *1#year^-1
            return etas

        if isoangle is not None:
            forceCalculate=False
            sigmaP = self.cross_section * (self.mu_XP(mX)/self.mu_Xe(mX))**2

            if FDMn == 0:
                fdm_str = 'Scr'
            else:
                fdm_str = 'LM'
            if useVerne:
                dir = f'./halo_data/modulated/Verne_{fdm_str}/'
            else:
                dir = f'./halo_data/modulated/Parameter_Scan_{fdm_str}/'
            directories = os.listdir(dir)
            masses = []
            cross_sections = []
            masses_sigmaE = []
            sigmaEs = []
            for d in directories:
                if 'Store' in d:
                    continue
                if 'sigmaE' in d:
                    mass = re.findall('DM_.*_MeV',d)[0][3:-4]
                    mass = mass.replace('_','.')
                    cross_section = re.findall('E_.*cm',d)[0][2:-3].replace('_','.')
                    mass = float(mass)
                    cross_section = float(cross_section)
                    masses_sigmaE.append(mass)
                    sigmaEs.append(cross_section)
                elif 'sigmaP' in d:    
                    mass = re.findall('DM_.*_MeV',d)[0][3:-4]
                    mass = mass.replace('_','.')
                    cross_section = re.findall('P_.*cm',d)[0][2:-3].replace('_','.')
                    mass = float(mass)
                    cross_section = float(cross_section)
                    masses.append(mass)
                    cross_sections.append(cross_section)
  
            from numpy import sort,unique
            cross_sections = sort(unique(cross_sections))
            masses = unique(masses)

            sigmaEs = sort(unique(sigmaEs))
            masses_sigmaE = unique(masses_sigmaE)
            # sigmaP_closest = self.findClosest(cross_sections, len(cross_sections), sigmaP)
            # discrepancy = sigmaP / sigmaP_closest
            # sigmaP_closest = str(sigmaP_closest)
            # sigmaP_closest = sigmaP_closest.replace('.','_')
            file = f'{dir}mDM_{mass_string}_MeV_sigmaE_{self.cross_section}_cm2/DM_Eta_theta_{isoangle}.txt'
            if not os.path.isfile(file):
                # print(useVerne,dir)
                print(file)
                raise FileNotFoundError('sigmaE file not found')
                # # print(mass_string)
                # # if float(mass_string).is_integer():
                # #     mass_string = mX*1e-6 #turn into MeV
                # #     mass_string = float(mass_string)
                # #     mass_string = npround(mass_string,3)
                # #     mass_string = int(mass_string)
                # file = f'{dir}mDM_{mass_string}_MeV_sigmaP_{sigmaP_closest}_cm2/DM_Eta_theta_{isoangle}.txt'
                # print('sigmaP File')
                # print(file)
            if npround(mX*1e-6,2) not in masses and npround(mX*1e-6,2) not in masses_sigmaE:
                print(mX*1e-6,masses,mass_string)
                raise ValueError(f"The mass you specified does not have a modulated eta file for this FDM. This must be generated with DaMasCus.\n Tried {file}")
            
            # file = f'{dir}mDM_{mass_string}_MeV_sigmaP_{sigmaP_closest}_cm2/DM_Eta_theta_{isoangle}.txt'
                # data = f.readlines()
                # for d in data:
                #     temp = d.split()
                #     file_etas.append(float(temp[1]))
                #     file_vmins.append(float(temp[0]))

        elif halo_model == 'imb':
            vMins = self.DM_Halo.vmin_tensor(Ee_tensor,q_tensor,mX)
            etas = self.DM_Halo.eta_MB_tensor(vMins) #s /cm
            etas *=ccms**2*sec2yr*self.rhoX/mX * self.cross_section
            return etas #year^-1
        elif forceCalculate:

            vMins = self.DM_Halo.vmin_tensor(Ee_tensor,q_tensor,mX)
            etas = torch.zeros_like(vMins)
            for i in range(etas.shape[0]):
                for j in range(etas.shape[1]):
                    v = vMins[i,j]
                    if v > (self.vEscape+self.vEarth)*2:
                        etas[i,j] = 0
                        continue
                    if halo_model == 'shm':
                        etas[i,j] = self.DM_Halo.etaSHM(v)
                    elif halo_model == 'tsa':
                        etas[i,j] = self.DM_Halo.etaTsa(v)
                    elif halo_model == 'dpl':
                        etas[i,j] = self.DM_Halo.etaDPL(v)
            etas*=ccms**2*sec2yr*self.rhoX/mX * self.cross_section#year^-1
            return etas
            

        else:
            try:
                halo_file = f'./halo_data/{halo_model}/mDM_{mass_string}_MeV.txt'
                temp =open(halo_file,'r')
                temp.close()
            except FileNotFoundError:
                #this probably just needs to be created
                raise FileNotFoundError(f'please generate halo file.\n Tried {halo_file}')
        

                # generate_halo_files(mX,crosssection,halo_model,Emin,Emax,dE)
            file = f'./halo_data/{halo_model}/mDM_{mass_string}_MeV.txt'
            # print(f'found halo file: {file}')
        from numpy import loadtxt
        try:
            data = loadtxt(file,delimiter='\t')
        except ValueError:
            print(file)
        if len(data) == 0:
            vMins = self.DM_Halo.vmin_tensor(Ee_tensor,q_tensor,mX)
            return torch.zeros_like(vMins)
        
        file_etas = data[:,1]
        file_vmins = data[:,0]
        if calcErrors == 'High':
            file_eta_err = data[:,2]
            file_etas += file_eta_err
        if calcErrors == 'Low':
            file_eta_err = data[:,2]
            file_etas -= file_eta_err
            # with open(f'./halo_data/{halo_model}/mDM_{mass_string}_MeV_sigmaE_{sigmaE_str}_cm2.txt','r') as f:
            #     data = f.readlines()
                
            #     if len(data) == 0:
            #         etas = np.zeros_like(q_array)
            #         return etas
            #     for d in data:
            #         temp = d.split()
            #         file_etas.append(float(temp[1]))
            #         file_vmins.append(float(temp[0]))
        
        
        # file_etas = np.array(file_etas)
        # file_vmins = np.array(file_vmins)

        eta_func = Akima1DInterpolator(file_vmins,file_etas)
        vMins = self.DM_Halo.vmin_tensor(Ee_tensor,q_tensor,mX)
        if isoangle is not None:
            from QEDarkConstants import lightSpeed_kmpers
            vMin_conversion_factor = lightSpeed_kmpers
            eta_conversion_factor  = 1e-5 #km/cm 
        else:
            vMin_conversion_factor = 1
            eta_conversion_factor  = 1
            #convert vMins to mattch eta file units


        vMins *= vMin_conversion_factor 
        # etas = np.zeros_like(vMins)

        # vMins = 
        # vMins = np.sort(vMins)
        # etas = np.interp(vMins,file_vmins,file_etas)
        vMin_numpy = vMins.cpu().numpy()
        etas = eta_func(vMin_numpy) # s/km
        etas = torch.from_numpy(etas)
        if self.device == 'mps':
            etas = etas.float()
        etas = etas.to(self.device)
        etas*=eta_conversion_factor #s/cm
        etas*=ccms**2*sec2yr*self.rhoX/mX * self.cross_section#year^-1

        # if isoangle is not None:
        #     print(f'sigmaP did not match exactly calculated: {sigmaP}\n found {sigmaP_closest}\n multiplying eta by discrepancy = {discrepancy}')
        # etas*=discrepancy

        #make sure to avoid interpolation issues where there isn't data
        etas = torch.where((vMins<file_vmins[0]) | (vMins > file_vmins[-1]) | (torch.isnan(etas)) ,0,etas)

        #file function should already be converted to s/cm


        
        return etas
        
    def p100_function(self,x_values,ne,material):
        from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator 
        from numpy import loadtxt
        import torch
        from copy import deepcopy

        
        p100data = loadtxt('./p100K.dat')
        # pEV_cut = 1000
        pEV = p100data[:,0]
        probs = p100data[:,:]
        probs = probs.T
        # index = np.where(pEV in x_values)[0]
        probs = probs[ne]
        p100_func = Akima1DInterpolator(pEV,probs)
        # x_temp = deepcopy(x_values.numpy())
        y_values = p100_func(x_values)
        y_values = torch.from_numpy(y_values)
        if self.device == 'mps':
            y_values=y_values.float()

        y_values = y_values.to(self.device)
        


        # pEV = pEV[::2]
        # probs = probs[::2]


        
        # x_temp = x_temp.astype(pEV.dtype)
        # x_temp = np.round(x_temp,2)
        # y_values = np.zeros_like(x_temp)



        # indices = np.argwhere(np.in1d(x_temp,pEV)).T
        # y_values[indices] = probs[indices]
        # y_values = torch.from_numpy(y_values)
        return y_values

    def step_function(self,x_values,ne,material):
        from QEDarkConstants import materials
        import torch
        # from copy import deepcopy
        start = materials[material][2]
        y_values = torch.zeros_like(x_values,device=self.device)
        bin_size =  materials[material][3]
        # print(bin_size)
        # print('start',start)
        # print(ne*bin_size + start)
        n_max = ne*bin_size + start 
        n_min = n_max - bin_size
        
        n_min = start + (ne-1)*bin_size
        m_max = n_min + bin_size -0.1
        y_values = torch.where((x_values < n_max) & (x_values >= n_min),1,0)
        # y_values = y_values.to(self.device)
        return y_values



    def vectorized_dRdE(self,material, mX, Ee_array, FDMn,halo_model,DoScreen=False,isoangle=None,halo_id_params=None,forceCalculate=False,useVerne=False,calcError=None):
        import torch
        from QEDarkConstants import materials,ccms,sec2yr,alpha,me_eV
        # print(halo_id_params,'just passing them in to see (dRdE)')
        q_array = torch.arange(1,self.nq+1,device=self.device)
        q_array_denom = torch.clone(q_array) * self.dQ
        etas = self.get_halo_data(mX,q_array,Ee_array,FDMn,halo_model,isoangle=isoangle,halo_id_params = halo_id_params,forceCalculate=forceCalculate,useVerne=useVerne,calcErrors=calcError)
        # print(etas,'before requiring grad')
        # etas.requires_grad_()
        # print(etas,'after requiring grad')

        Eprefactor = materials[material][1]
        Mcell = materials[material][0]
       
        
        prefactor_crys = 1/Mcell*alpha*me_eV**2 / self.mu_Xe(mX)**2 
        if self.device != 'mps':
            Ei_array = torch.floor(torch.round(Ee_array*10)).int()
        else:
            from numpy import round as npround
            Ee_array_temp = Ee_array.cpu().numpy()

            Ei_array = torch.floor(torch.tensor(npround(Ee_array_temp*10,2))).int()


        Ei_array = Ei_array.cpu().numpy()
        q_array_denom_tiled= torch.tile(q_array_denom,(len(Ee_array),1))




        extra_factor_test = torch.ones_like(q_array,device=self.device)#1 - (1/(2*q_array))
        extra_factor_test_tiled = torch.tile(extra_factor_test,(len(Ee_array),1))




        # Ee_array_tiled = np.tile(Ee_array,(len(q_arr),1)).T
        fdm_tiled = self.FDM(q_array_denom_tiled,FDMn)
        # print(q_array_denom_tiled.shape,fdm_tiled.shape,etas.shape,materials['Si'][4][:,Ei_array].shape,TFscreening(q_array_denom, Ee_array, DoScreen).shape)
        material_arr = materials[material][4][:,Ei_array-1]
        material_arr = material_arr.T
        material_arr = torch.from_numpy(material_arr)
        if self.device == 'mps':
            material_arr = material_arr.float()
            q_array_denom_tiled = q_array_denom_tiled.float()
            etas = etas.float()
            fdm_tiled = fdm_tiled.float()

        material_arr = material_arr.to(self.device)


        
        # print(type(fdm_tiled),type(q_array_denom_tiled),type(etas),type(material_arr))
        # if DoScreen:
        tf_factor = (self.TFscreening(q_array_denom, Ee_array, DoScreen)**2)
        # else:
            # tf_factor = 1
        result = prefactor_crys*Eprefactor*(1/q_array_denom_tiled)*etas*fdm_tiled**2*material_arr*tf_factor*extra_factor_test_tiled
        # wrong_index = np.where(result != 0)
        # print(wrong_index[0],result[wrong_index],1/q_array_denom[wrong_index],etas[wrong_index],materials['Si'][4][wrong_index[0],Ei-1])
        result= torch.sum(result,axis=1)
        # print(result,'this result needs to maintain requires_grad')
        # print(result)
        band_gap_result = torch.where(Ee_array < materials[material][2],0,result)
        return band_gap_result#,[prefactor_crys,Eprefactor,(1/q_array_denom_tiled),etas,fdm_tiled**2,material_arr]



    def vectorized_dRdnE(self,material,mX,ne,FDMn,halo_model,DoScreen=False,isoangle=None,halo_id_params=None,forceCalculate=False,useVerne=False,calcError=None):
        #possible options for calcError None, High, Low
        import torch
        import numpy
        #takes in mass in MeV
        from QEDarkConstants import materials



        if material == 'Ge':
            if self.ionization_func is not self.step_function:
                self.change_to_step()
        # self.update_Emin(materials[material][2])




        if type(ne) != torch.Tensor:
            if type(ne) == int:
                nes = torch.tensor([ne],device=self.device)
            elif type(ne) == list:
                nes = torch.tensor(ne,device=self.device)
            elif type(ne) == numpy.ndarray:
                nes = torch.from_numpy(ne)
                nes = nes.to(self.device)
            else:
                try:
                    nes = torch.tensor(ne,device=self.device)
                except:
                    print('unknown data type')
    

        else:
            nes = ne

        

        Ee_array = torch.arange(self.Emin,self.Emax,step = self.dE,device=self.device)
        if self.device != 'mps':
            Ee_array=torch.round(Ee_array,decimals=2)

        fn_tiled = []
        for ne in nes:
            temp = self.ionization_func(Ee_array,ne,material)
            temp = torch.where(torch.isnan(temp),0,temp)
            fn_tiled.append(temp)
        fn_tiled = torch.stack(fn_tiled)



        dRdE = self.vectorized_dRdE(material, mX*1e6, Ee_array, FDMn, halo_model,DoScreen=DoScreen,isoangle=isoangle,halo_id_params=halo_id_params,forceCalculate=forceCalculate,useVerne=useVerne,calcError=calcError)
        # print(dRdE)
        # dRdE_tiled = torch.tile(dRdE,(len(nes),1))
        tmpdRdE = dRdE*fn_tiled
        
        dRdne = torch.sum(tmpdRdE, axis = 1)

        return dRdne#,dRdE,fn_tiled,Ee_array,tmpdRdE #tensor


    # def vectorized_dRdnE(material,mX,ne,FDMn,halo_model,ionization_function,DoScreen=False,isoangle=None,halo_id_params=None):
    #     from QEDarkConstants import Emin,Emax,dE
    #     import torch
    #     # print(halo_id_params)
    #     Ee_array = torch.arange(Emin,Emax,step = dE)
    #     Ee_array=torch.round(Ee_array,decimals=2)
    #     fn_values = ionization_function(Ee_array,ne,material)
    #     # Ee_array = torch.where(fn_values > 0,Ee_array,0)
    #     # fn_values = torch.where(fn_values > 0,fn_values,0)
    #     dRdE = vectorized_dRdE(material, mX, Ee_array, FDMn, halo_model,DoScreen=DoScreen,isoangle=isoangle,halo_id_params=halo_id_params)
    #     tmpdRdE = dRdE*fn_values
        
    #     dRdne = torch.sum(tmpdRdE, axis = 0)
    #     # print(dRdne,'now Drdne needs to maintain requires_grad')


    #     return dRdne




    def generate_dat(self,dm_masses,ne_bins,fdm,dm_halo_model,material,DoScreen=False,write=True):
        from tqdm.notebook import tqdm_notebook
        from QEDarkConstants import lightSpeed_kmpers
        import numpy as np
        function_name = str(self.ionization_func).strip('<<bound method QEDark.')[:4]
        fdm_dict = {0:'1',1:'q',2:'q2'}
        #FDM1_vesc544-v0220-vE232-Ebin3pt8-rhoX0pt4_nevents.dat
        from QEDarkConstants import materials
        rho_X =self.rhoX
        
        vE = self.vEarth 
        vesc = self.vEscape   
        v0 = self.v0
        rho_X_print = str(np.round(rho_X*1e-9,1))
        ebin_print = str(materials[material][3])
        vesc_print = str(np.round(vesc,1))
        v0_print = str(np.round(v0,1))
        vE_print = str(np.round(vE,1))


        ebin_print = ebin_print.replace('.','pt')
        rho_X_print = rho_X_print.replace('.','pt')
        vesc_print = vesc_print.replace('.','pt')
        v0_print = v0_print.replace('.','pt')
        vE_print = vE_print.replace('.','pt')


        
        FDM_Dir = '../../../DarkMatterRates/QEDark/'
        if DoScreen:
            screen ='_screened'
        else:
            screen = ''
        filename = FDM_Dir + f'FDM{fdm_dict[fdm]}_vesc{vesc_print}-v0{v0_print}-vE{vE_print}-Ebin{ebin_print}-rhoX{rho_X_print}_nevents_func{function_name}_maxne{np.max(ne_bins)}_unscaled{screen}_qedark3.dat'
        
        lines = []
        data = np.zeros((len(dm_masses),len(ne_bins)))
        for m in tqdm_notebook(range(len(dm_masses))):
            mX = dm_masses[m]
            line = f'{mX}\t'
            kg_year = self.vectorized_dRdnE(material, mX, ne_bins, fdm, dm_halo_model,DoScreen=DoScreen,isoangle=None)
            g_year = kg_year / 1000
            g_day = g_year * (1/365)
            g_day = g_day.numpy()
            for ne in ne_bins:
                data[m,ne - 1] = g_day[ne-1]
                line+=str(g_day[ne-1])+'\t'
                
            line += f'\n'
            lines.append(line)
        if write: 
            f = open(filename,'w')
            first_line = f'FDM{fdm_dict[fdm]}:\tmX [MeV]\tsigmae={self.cross_section} [cm^2]\t'
            for ne in ne_bins:
                first_line += f'ne{ne}\t'
            first_line+='\n'
            f.write(first_line)
            for line in lines:
                f.write(line)
        return data



    


     


