
import torch
import numpy as np
from numpy import loadtxt as loadtxt
import matplotlib.pyplot as plt
import pickle
import os

from tqdm import tqdm
from copy import deepcopy
from QEDark3 import QEDark
from QEDarkConstants import ccms,lightSpeed_kmpers,lightSpeed,sec2yr


epsilon = 1e-10

def log_likelihood(signal_rate, background, n_obs):

    log_part = n_obs * torch.log((n_obs/(signal_rate + background + epsilon)) + epsilon)
    log_likelihood = signal_rate + background - n_obs + log_part
    log_likelihood = 2 * torch.sum(log_likelihood)
    return log_likelihood  


def profile_likelihood(mX, params_to_fit, materials, FDMn, observed_rates, QE):
    
    # calculate expected rates given the parameters
    nes = torch.arange(1, len(observed_rates[0])+1, device=params_to_fit.device)
    likelihoods = 0
    
    for material_idx in range(len(materials)):
        signal = QE.vectorized_dRdnE(materials[material_idx],mX,nes,FDMn,'idp',DoScreen=False,isoangle=None,halo_id_params=params_to_fit)
        background = torch.zeros_like(signal,device=signal.device)#assume zero background
        likelihoods+=log_likelihood(signal, background, observed_rates[material_idx])

    return likelihoods # we want to maximize the likelihood


    
def calculate_chi_squared(mX,n_obs,material,FDMn,_params,QE):
    nes = torch.arange(1,len(n_obs)+1)
    signal = QE.vectorized_dRdnE(material,mX,nes,FDMn,'idp',DoScreen=False,isoangle=None,halo_id_params=_params)

    signal_epsilon = torch.where(signal ==0,epsilon,signal)
    background = torch.zeros_like(signal)#assume zero background
    chi_2 = ((n_obs - signal_epsilon - background)**2) / (signal_epsilon+background)
    
    return torch.sum(chi_2)



    
    
def convert_params_to_fit_to_halo_id(learnable_weights):
    return -torch.cumsum(learnable_weights**2, dim=0)

def minimize(mX, FDMn, observed_rates, materials, loss_function, params_to_fit, epochs=5, lr=0.001, clip_value=5, device='cpu', optimizer_algorithm='Adam', adaptive=True, fused=False, alpha=0.1):
    
    QE = QEDark()
    QE.change_to_step() #default to step
    QE.optimize(device)
    
    if device != 'cuda':
        fused = False
        
    list_params = []
    params_to_fit.requires_grad_()
    if fused:
        print('using fused implementation')
    if optimizer_algorithm == 'Adam':
        optimizer = torch.optim.Adam([params_to_fit], lr=lr,fused=fused)
    elif optimizer_algorithm == 'AdamW':
        optimizer = torch.optim.AdamW([params_to_fit], lr=lr,fused=fused)
    elif optimizer_algorithm == 'RAdam':
        optimizer = torch.optim.RAdam([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'SGD':
        optimizer = torch.optim.SGD([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'ASGD':
        optimizer = torch.optim.ASGD([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'Adamax':
        optimizer = torch.optim.Adamax([params_to_fit], lr=lr)
    # elif optimizer_algorithm == 'LBFGS':
    #     optimizer = torch.optim.LBFGS([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'NAdam':
        optimizer = torch.optim.NAdam([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'Adagrad':
        optimizer = torch.optim.Adagrad([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'Adadelta':
        optimizer = torch.optim.Adadelta([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'NAdam':
        optimizer = torch.optim.NAdam([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'RMSprop':
        optimizer = torch.optim.RMSprop([params_to_fit], lr=lr)
    elif optimizer_algorithm == 'Rprop':
        optimizer = torch.optim.Rprop([params_to_fit], lr=lr)

    losses = []
    if adaptive:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = loss_function(mX, convert_params_to_fit_to_halo_id(params_to_fit), materials, FDMn, observed_rates, QE) + alpha*monotonicity_constraint(params_to_fit)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(params_to_fit, clip_value)
        optimizer.step()
        if adaptive:
            scheduler.step(loss)
        if torch.isnan(loss):
            print(list_params)
            print('this loss function became nan')
            break
        list_params.append(deepcopy(params_to_fit))
        losses.append(float(loss))

    test_statistic = []
    for material_idx in range(len(materials)):
        test_stat = calculate_chi_squared(mX, observed_rates[material_idx],materials[material_idx],FDMn,convert_params_to_fit_to_halo_id(params_to_fit),QE)
        test_statistic.append(float(test_stat))
    
    return params_to_fit, losses, test_statistic


def minimize_with_mass(FDMn, observed_rates, materials, loss_function, params_to_fit, mass_to_fit, epochs=5, lr=0.001, clip_value=5, device='cpu', optimizer_algorithm='Adam', adaptive=True, fused=False, alpha=1):
    
    QE = QEDark()
    QE.change_to_step() #default to step
    QE.optimize(device)
    
    if device != 'cuda':
        fused = False
        
    params_to_fit.requires_grad_()
    mass_to_fit.requires_grad_()
    
    
    if fused:
        print('using fused implementation')
    if optimizer_algorithm == 'Adam':
        optimizer_g = torch.optim.Adam([params_to_fit], lr=lr,fused=fused)
        optimizer_mass = torch.optim.Adam([mass_to_fit], lr=alpha*lr,fused=fused)

    losses = []
    masses = []
    
    if adaptive:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min')
    for i in tqdm(range(epochs)):
        
        
        optimizer_g.zero_grad()
        optimizer_mass.zero_grad()
        
        loss = loss_function(mass_to_fit**2, convert_params_to_fit_to_halo_id(params_to_fit), materials, FDMn, observed_rates, QE)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params_to_fit, clip_value)
        torch.nn.utils.clip_grad_norm_(mass_to_fit, clip_value)

        optimizer_g.step()
        optimizer_mass.step()
        
        
        if adaptive:
            scheduler.step(loss)
    
        if torch.isnan(loss):
            print(list_params)
            print('this loss function became nan')
            break
            
        losses.append(float(loss))
        masses.append(float(mass_to_fit**2))
        

    return params_to_fit, losses, masses


def plot_eta(mX,params,plot_mb =True,device='cpu',save=False,plotname=None,dir='./',cross_section = 1e-36,halo_params=None,norm=1e-10):
    #halo_params = [238.0,250.2,544,0.3e9] <- acceptable format to input halo params v0,vE,vEsc,rhoX
    
    QE = QEDark()
    
    if halo_params is not None:
        QE.update_params(halo_params[0],halo_params[1],halo_params[2],halo_params[3],cross_section)
    else:
        QE.update_crosssection(cross_section)
    QE.optimize(device)
    
    q = np.arange(1,901)*QE.dQ
    vMinimum = np.min(2*1.2 /q + q / (2*mX*1e6))
    vMinimum*=lightSpeed_kmpers
    # print(vMinimum)
    vMins = torch.arange(1,1200,step=1200/2400,device=device)/lightSpeed_kmpers
    eta_shm = []
    step_heights = torch.exp(params).detach()
    num_steps = params.shape[0]
    vis = torch.arange(0,1000,step = 1000/num_steps,device=device)/lightSpeed_kmpers
    vis_tiled = torch.tile(vis,(vMins.shape[0],1))
    vMins_tiled = torch.tile(vMins,(vis.shape[0],1)).T
    step_heights_tiled = torch.tile(step_heights,(vMins.shape[0],1))

    temp = vis_tiled-vMins_tiled
    heaviside = torch.where(temp > 0, 1,0)
    vis_tiled.shape,vMins_tiled.shape,step_heights_tiled.shape
    etas =  torch.sum(step_heights_tiled * heaviside,axis=1).T

    fig,ax = plt.subplots(figsize=(15,10))

    vMins*=lightSpeed_kmpers
    vMins = vMins.cpu().numpy()
    crop = np.where(vMins > vMinimum)[0]

    if plot_mb:
        for v in vMins:
            v_c= v/lightSpeed_kmpers
            eta_shm.append(QE.DM_Halo.etaSHM(float(v_c)))

        eta_shm = np.array(eta_shm)*ccms**2*sec2yr*QE.rhoX/(mX*1e6) * QE.cross_section
        eta_shm = eta_shm[crop]
    etas = etas.cpu().numpy()*norm

    
    vMins = vMins[crop]
    etas = etas[crop]
    
    if plot_mb:
        ax.plot(vMins,eta_shm,label='SHM')
        ax.scatter(vMins,etas,label='Best Fit η',color='red')
    else:
        ax.plot(vMins,etas,label='Best Fit η')
    ax.legend()
    ax.set_ylabel('η (s/km)')
    ax.set_xlabel('Vmin (km/s)')

    fig.suptitle(f'Best Fit Eta Mass {mX} MeV',fontsize=32)
    plt.tight_layout()
    if save:
        if plotname is not None:
            figname = dir + plotname
        else:
            figname = dir+'eta_plot.png'
        plt.savefig(figname, bbox_inches='tight')

    return etas,eta_shm
       
def plot_eta_params(data,name,test_mX,FDMn,plot_shm = False,cross_section=1e-36,halo_params=None):
    likelihood_fdm0 = []
    test_stats_fdm0 = []
    likelihood_fdm2 = []
    test_stats_fdm2 = []
    params_fdm0 = []
    params_fdm2 = []
    dm_masses = list(data[0].keys())
    dm_masses = np.array(dm_masses)
    for mX in dm_masses:
        likelihood0 = data[0][mX]['likelihoods']
        likelihood2 = data[2][mX]['likelihoods']
        likelihood_fdm0.append(likelihood0)
        likelihood_fdm2.append(likelihood2)

        teststats0 = data[0][mX]['test_statistics']
        teststats2 = data[2][mX]['test_statistics']
        test_stats_fdm0.append(teststats0)
        test_stats_fdm2.append(teststats2)

        params0 = data[0][mX]['params']
        params2 = data[2][mX]['params']
        params_fdm0.append(params0)
        params_fdm2.append(params2)

    if FDMn == 2:
        params  = params_fdm2
    else:
        params = params_fdm0

    mass_index = np.where(dm_masses == test_mX)[0][0]
    print(dm_masses,test_mX)
    print(mass_index)
    best_fit_mass = dm_masses[mass_index]
    best_fit_params = params[mass_index]
    param_device =best_fit_params.device
    test_eE = 1
    eta_name = f'{name}_FDM{FDMn}_mX_{best_fit_mass}_eta.png'

    plot_eta(best_fit_mass,best_fit_params,plot_mb =plot_shm,device=param_device,save=False,plotname=eta_name,dir='./',cross_section=cross_section,halo_params=halo_params)


def find_best_fit(data_name,model_name,plot_shm = True,zoom=True,log=True,cross_section=1e-36,onlySi=False,halo_params=None,save=False):
    
    savedir='./halo_independent/halo_idp_plots/'
    #given a data file from a model scan above, 
    #make sure likelihoods converged
    #plot the test statistics
    #find the best fit mass, params,
    #plot etas

    with open(f'./halo_independent/halo_idp_results/{data_name}','rb') as dbfile:
        data = pickle.load(dbfile)

    likelihood_fdm0 = []
    test_stats_fdm0 = []
    likelihood_fdm2 = []
    test_stats_fdm2 = []
    params_fdm0 = []
    params_fdm2 = []
    dm_masses = list(data[0].keys())
    print(dm_masses)

    for mX in dm_masses:
        likelihood0 = data[0][mX]['likelihoods']
        likelihood2 = data[2][mX]['likelihoods']
        likelihood_fdm0.append(likelihood0)
        likelihood_fdm2.append(likelihood2)

        teststats0 = data[0][mX]['test_statistics']
        teststats2 = data[2][mX]['test_statistics']
        test_stats_fdm0.append(teststats0)
        test_stats_fdm2.append(teststats2)

        params0 = data[0][mX]['params']
        params2 = data[2][mX]['params']
        params_fdm0.append(params0)
        params_fdm2.append(params2)

    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(len(likelihood_fdm0)):
        ax.plot(likelihood_fdm0[i][-1000:],label=f'{dm_masses[i]}')
    ax.set_xlabel('Epochs')
    # ax.set_ylim([-2,1e6])
    ax.set_ylabel("Likelihood")
    if log:
        ax.set_yscale('log')
    fig.suptitle(f'Likelihoods FDM0 {model_name}',fontsize=32)
    plt.tight_layout()
    ax.legend()
    if save:
        plt.savefig(f'{savedir}LikelihoodsFDM0_{model_name}.png', bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(len(likelihood_fdm2)):
        ax.plot(likelihood_fdm2[i][-1000:],label=f'{dm_masses[i]}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel("Likelihood")
    # ax.set_ylim([-2,1e6])
    if log:
        ax.set_yscale('log')
    fig.suptitle(f'Likelihoods FDM2 {model_name}',fontsize=32)
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f'{savedir}LikelihoodsFDM2_{model_name}.png', bbox_inches='tight')




    test_stats_fdm0 = np.array(test_stats_fdm0)
    combined_fdm0 = test_stats_fdm0[:,0] + test_stats_fdm0[:,1]
    fig,ax = plt.subplots(figsize=(15,10))
    ax.plot(dm_masses,test_stats_fdm0[:,0],label='Si')
    if not onlySi:
        ax.plot(dm_masses,test_stats_fdm0[:,1],label='Ge')
        ax.plot(dm_masses,combined_fdm0,label='Combined')
    #
    ax.set_xscale('log')
    ax.set_xlabel('DM Mass [MeV]')
    ax.set_ylabel('$\chi^2$')
    if zoom:
        ax.set_ylim([0,100])
        ax.set_xlim([1,1000])
    else:
        ax.set_yscale('log')
    ax.legend()

    fig.suptitle(f"χ$^2$ FDM0 {model_name}",fontsize=32)
    plt.tight_layout()
    plt.savefig(f'{savedir}Chi2FDM0_{model_name}.png', bbox_inches='tight')
    print('*********************************************')
    print('BEST FIT MASSES AND χ^2 FDM0')
    print('*********************************************')

    print(dm_masses[np.nanargmin(test_stats_fdm0[:,0])],np.nanmin(test_stats_fdm0[:,0]),'Si')
    if not onlySi:
        print(dm_masses[np.nanargmin(test_stats_fdm0[:,1])],np.nanmin(test_stats_fdm0[:,1]),'Ge')
        print(dm_masses[np.nanargmin(combined_fdm0)],np.nanmin(combined_fdm0),'Combined')
    print()
    print()
    print()



    test_stats_fdm2 = np.array(test_stats_fdm2)
    
    fig,ax = plt.subplots(figsize=(15,10))
    ax.plot(dm_masses,test_stats_fdm2[:,0],label='Si')
    if not onlySi:
        combined_fdm2 = test_stats_fdm2[:,0] + test_stats_fdm2[:,1]
        ax.plot(dm_masses,test_stats_fdm2[:,1],label='Ge')
        ax.plot(dm_masses,combined_fdm2,label='Combined')
    
    ax.set_xscale('log')
    ax.set_xlabel('DM Mass [MeV]')
    ax.set_ylabel('$\chi^2$')
    if zoom:
        ax.set_ylim([0,100])
        ax.set_xlim([1,1000])
    else:
        ax.set_yscale('log')
    ax.legend()

    fig.suptitle(f"χ$^2$ FDM2 {model_name}",fontsize=32)
    plt.tight_layout()
    if save:
        plt.savefig(f'{savedir}Chi2FDM2_{model_name}.png', bbox_inches='tight')

    print('*********************************************')
    print('BEST FIT MASSES AND χ^2 FDM2')
    print('*********************************************')

    print(dm_masses[np.nanargmin(test_stats_fdm2[:,0])],np.nanmin(test_stats_fdm2[:,0]),'Si')
    if not onlySi:
        print(dm_masses[np.nanargmin(test_stats_fdm2[:,1])],np.nanmin(test_stats_fdm2[:,1]),'Ge')
        print(dm_masses[np.nanargmin(combined_fdm2)],np.nanmin(combined_fdm2),'Combined')
    print()
    print()
    print()

    if onlySi:
        comparisonFDM2 = test_stats_fdm2[:,0]
        comparisonFDM0 = test_stats_fdm0[:,0]
    else:
        comparisonFDM2 = combined_fdm2
        comparisonFDM0 = combined_fdm0

    # if np.min(comparisonFDM2) < np.min(comparisonFDM0):
        # best_fdm = 2
    best_index_fdm0 = np.nanargmin(comparisonFDM2)
    name_index_2 = 2
# else:
    # best_fdm = 0
    best_index_fdm2 = np.nanargmin(comparisonFDM0)
    name_index_0 = 0


    best_fit_mass_fdm0 = dm_masses[best_index_fdm0]
    best_fit_params_fdm0 = params_fdm0[best_index_fdm0]
    param_device_fdm0 =best_fit_params_fdm0.device
    eta_name = f'{model_name}_FDM{name_index_0}_mX_{best_fit_mass_fdm0}_best_fit_eta.png'
    plot_eta(best_fit_mass_fdm0,best_fit_params_fdm0,plot_mb =plot_shm,device=param_device_fdm0,save=save,plotname=eta_name,dir=savedir,cross_section=cross_section,halo_params=halo_params) 
        
    best_fit_mass_fdm2 = dm_masses[best_index_fdm2]
    best_fit_params_fdm2 = params_fdm2[best_index_fdm2]
    param_device_fdm2 =best_fit_params_fdm2.device
    eta_name = f'{model_name}_FDM{name_index_2}_mX_{best_fit_mass_fdm2}_best_fit_eta.png'
        
    plot_eta(best_fit_mass_fdm2,best_fit_params_fdm2,plot_mb =plot_shm,device=param_device_fdm2,save=save,plotname=eta_name,dir=savedir,cross_section=cross_section,halo_params=halo_params) 


    return data


    

def mock_data_scan(modelnumber,device,masses=None,onlySi=False,adaptive=False,mock_data_dir='1kgyr_fake_data',optimizer='AdamW'):

    print(f'looking at files from {mock_data_dir}')
    fpath_si = f'./halo_independent/{mock_data_dir}/Model{modelnumber}_Si.csv'
    fake_data_si = loadtxt(fpath_si,delimiter=',')
    rates_si = fake_data_si[:,1]
    rates_si = torch.from_numpy(rates_si)
    if device == 'mps':
        rates_si = rates_si.float()
    rates_si = rates_si.to(device)
    fpath_ge = f'./halo_independent/mock_data/Model{modelnumber}_Ge.csv'
    fake_data_ge = loadtxt(fpath_ge,delimiter=',')
    rates_ge = fake_data_ge[:,1]
    rates_ge = torch.from_numpy(rates_ge)
    if device == 'mps':
        rates_ge = rates_ge.float()
    rates_ge = rates_ge.to(device)
    model_rates = [rates_si,rates_ge]
    materials = ['Si','Ge']
    if masses is None:
        dm_masses = torch.concatenate((torch.tensor([1,3]),torch.arange(5,100,step=5),torch.arange(100,1100,step=100))).numpy()
    else:
        dm_masses = masses

    if onlySi:
        data = model_scan(dm_masses, [rates_si],['Si'],num_steps = 100,lr=0.001,epochs=15000,device=device,model_name=f'model{modelnumber}',adaptive=adaptive,optimizer=optimizer)
        picklename = f'halo_independent/halo_idp_results/model{modelnumber}_{device}_Si.pickle'
    else:
        data = model_scan(dm_masses,model_rates,materials,num_steps = 100,lr=0.001,epochs=15000,device=device,model_name=f'model{modelnumber}',adaptive=adaptive,optimizer=optimizer)
        picklename = f'halo_independent/halo_idp_results/model{modelnumber}_{device}.pickle'
    if os.path.isfile(picklename):
        os.system(f'rm {picklename}')

    dbfile = open(picklename, 'ab')
     
    # source, destination
    pickle.dump(data, dbfile)                    
    dbfile.close()
    return data



def model_scan(dm_masses,model_rates,materials,num_steps = 100,lr=0.001,epochs=1e5,device='cpu',model_name='ModelName',adaptive=False,optimizer='Adam'):
  
    FDMs = [0,2]
    data_dict = {}
    data_dict[FDMs[0]] = {}
    data_dict[FDMs[1]] = {}

    for FDMn in FDMs:
        for m in tqdm(range(len(dm_masses))):
            mX = dm_masses[m]
            mX_dict = {}
            fpath = f'./halo_independent/halo_idp_results/temp/{model_name}_{device}_mX{mX}_FDMn{FDMn}.pickle'
            if os.path.isfile(fpath):
                dbfile = open(fpath, 'rb') 
                mX_dict = pickle.load(dbfile)
                data_dict[FDMn][mX] = mX_dict
            else:
                if device == 'mps':
                    dtype=torch.float
                else:
                    dtype=torch.double
                test_params= torch.rand(num_steps,dtype=dtype,device=device)
                params_,likeilhoods_,test_stats_ =  minimize(mX,FDMn,model_rates,materials,profile_likelihood, test_params,epochs=int(epochs),lr=lr,device=device,adaptive=adaptive,optimizer_algorithm=optimizer)
                mX_dict['params'] = params_
                mX_dict['likelihoods'] = likeilhoods_
                mX_dict['test_statistics'] = test_stats_
                data_dict[FDMn][mX] = mX_dict
                dbfile = open(fpath, 'ab')
                pickle.dump(mX_dict, dbfile)
                dbfile.close()


        #delete temp files created
    os.system(f'rm ./halo_independent/halo_idp_results/temp/{model_name}_{device}*.pickle')
    return data_dict


if __name__ == "__main__":
    model = 4
    fpath_si = f'./halo_independent/mock_data/Model{model}_Si.csv'
    
    device = 'mps'
    fake_data_si = loadtxt(fpath_si,delimiter=',')
    rates_si = fake_data_si[:,1]
    rates_si = torch.from_numpy(rates_si)
    if device == 'mps':
        rates_si = rates_si.float()
    rates_si = rates_si.to(device)
    fpath_ge = f'./halo_independent/mock_data/Model{model}_Ge.csv'
    fake_data_ge = loadtxt(fpath_ge,delimiter=',')
    rates_ge = fake_data_ge[:,1]
    rates_ge = torch.from_numpy(rates_ge)
    if device == 'mps':
        rates_ge = rates_ge.float()
    rates_ge = rates_ge.to(device)
    model_rates = [rates_si,rates_ge]
    materials = ['Si','Ge']
    dm_masses = [1,5,10,50,100,1000]


    data = model_scan(dm_masses,model_rates,materials,num_steps = 100,lr=0.001,epochs=10000,cross_section=1e-37,device=device)

    dbfile = open(f'model{model}', 'ab')
     
    # source, destination
    pickle.dump(data, dbfile)                    
    dbfile.close()
