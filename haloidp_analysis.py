
import torch




def log_likelihood(signal_rate,background,n_obs):
    # log_likelihood = 0

    epsilon = torch.tensor(1e-5,dtype=signal_rate.dtype,device=signal_rate.device)
    signal_temp = signal_rate + epsilon
    inside_log = n_obs / (signal_temp+background)
    inside_log_temp = torch.where(inside_log==0,1,inside_log)
    inside_log_temp = torch.where(torch.isclose(signal_temp,epsilon,rtol=1e-5),1,inside_log_temp)
    log_part = n_obs * torch.log(inside_log_temp)

    log_likelihood = signal_rate + background - n_obs + log_part


    log_likelihood = 2 * torch.sum(log_likelihood)


    return log_likelihood  

def profile_likelihood(mX,params,materials,FDMn,n_obs_m,QE):
    nes = torch.arange(1,len(n_obs_m[0])+1,device=params.device)
    likelihoods = 0
    for m in range(len(n_obs_m)):
        signal = QE.vectorized_dRdnE(materials[m],mX,nes,FDMn,'idp',DoScreen=False,isoangle=None,halo_id_params=params)
        background = torch.zeros_like(signal,device=signal.device)#assume zero background
        likelihoods+=log_likelihood(signal,background,n_obs_m[m])

    return likelihoods
      





    
def calculate_chi_squared(mX,n_obs,material,FDMn,_params,QE,epsilon=1e-10):
    nes = torch.arange(1,len(n_obs)+1)
    signal = QE.vectorized_dRdnE(material,mX,nes,FDMn,'idp',DoScreen=False,isoangle=None,halo_id_params=_params)

    signal_epsilon = torch.where(signal ==0,epsilon,signal)
    background = torch.zeros_like(signal)#assume zero background
    chi_2 = ((n_obs - signal_epsilon - background)**2) / (signal_epsilon+background)
    
    return torch.sum(chi_2)




def minimize(mX,FDMn,n_obs_m,materials,function,initial_parameters,epochs=5,lr=0.001,clip_value=5,device='cpu',optimizer_algorithm='Adam',adaptive=True,fused=False):
    from tqdm.autonotebook import tqdm
    from copy import deepcopy
    from QEDark3 import QEDark
    QE = QEDark()
    QE.change_to_step() #default to step
    QE.optimize(device)
    if device != 'cuda':
        fused = False
    list_params = []
    params = initial_parameters
    params.requires_grad_()
    if fused:
        print('using fused implementation')
    if optimizer_algorithm == 'Adam':
        optimizer = torch.optim.Adam([params], lr=lr,fused=fused)
    elif optimizer_algorithm == 'AdamW':
        optimizer = torch.optim.AdamW([params], lr=lr,fused=fused)
    elif optimizer_algorithm == 'RAdam':
        optimizer = torch.optim.RAdam([params], lr=lr)
    elif optimizer_algorithm == 'SGD':
        optimizer = torch.optim.SGD([params], lr=lr)
    elif optimizer_algorithm == 'ASGD':
        optimizer = torch.optim.ASGD([params], lr=lr)
    elif optimizer_algorithm == 'Adamax':
        optimizer = torch.optim.Adamax([params], lr=lr)
    # elif optimizer_algorithm == 'LBFGS':
    #     optimizer = torch.optim.LBFGS([params], lr=lr)
    elif optimizer_algorithm == 'NAdam':
        optimizer = torch.optim.NAdam([params], lr=lr)
    elif optimizer_algorithm == 'Adagrad':
        optimizer = torch.optim.Adagrad([params], lr=lr)
    elif optimizer_algorithm == 'Adadelta':
        optimizer = torch.optim.Adadelta([params], lr=lr)
    elif optimizer_algorithm == 'NAdam':
        optimizer = torch.optim.NAdam([params], lr=lr)
    elif optimizer_algorithm == 'RMSprop':
        optimizer = torch.optim.RMSprop([params], lr=lr)
    elif optimizer_algorithm == 'Rprop':
        optimizer = torch.optim.Rprop([params], lr=lr)

    losses = []
    if adaptive:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = function(mX,params,materials,FDMn,n_obs_m,QE)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, clip_value)
        optimizer.step()
        if adaptive:
            scheduler.step(loss)
        if torch.isnan(loss):
            print(list_params)
            print('this loss function became nan')
            break
        list_params.append(deepcopy(params))
        losses.append(float(loss))
        # if i > nelements_to_check:
        #     derivs = torch.diff(torch.tensor(losses))
        #     derivs = derivs[-nelements_to_check:]
        #     derivs_check = torch.where(torch.abs(derivs) < 5, 0, derivs)
        #     if torch.sum(derivs_check) == 0:
        #         print(f'converged at {i} epochs')
        #         break
    test_statistic = []
    for m in range(len(materials)):
        test_stat = calculate_chi_squared(mX,n_obs_m[m],materials[m],FDMn,params,QE)
        test_statistic.append(float(test_stat))
    return params,losses,test_statistic





# def plot_eta(mX,params,plot_mb =True,device='cpu',save=False,plotname=None,dir='./'):

#     import matplotlib.pyplot as plt

#     import numpy as np
#     from QEDark3 import QEDark
#     from QEDarkConstants import ccms,lightSpeed_kmpers,lightSpeed
#     QE = QEDark()
#     QE.optimize(device)

#     q_tensor = torch.arange(1,QE.nq+1)*QE.dQ
#     eE_tensor = torch.arange(QE.Emin,QE.Emax,step=QE.dE)
#     eE_array = np.arange(QE.Emin,QE.Emax,QE.dE)
#     test_eE = 2
#     vMax = 1000 / lightSpeed_kmpers
#     num_steps = params.shape[0]
#     vMins = QE.DM_Halo.vmin_tensor(eE_tensor,q_tensor,mX*1e6) #units of c
#     vis = torch.arange(0,vMax,step = vMax/num_steps,device=params.device)
#     vis_np = np.arange(0,vMax,vMax/num_steps) * lightSpeed_kmpers
#     etas = torch.exp(params)*1e5#/lightSpeed*1e4#units of s/cm
#     if etas.device != 'cpu':
#         etas = etas.to('cpu')
#     etas_np = etas.detach().numpy()


#     fig,ax = plt.subplots(figsize=(15,10))
#     check_index = np.where(np.isclose(eE_array,test_eE))[0][0]
#     vMins_indexed = vMins[check_index,:]

#     ax.scatter(vis_np,etas_np ,label='Best Fit η')
#     if plot_mb:
#         etas_mb = []
#         for v in vMins_indexed:
#             vtest = float(v)
#             etas_mb.append(QE.DM_Halo.etaSHM(vtest))
#         etas_mb = np.array(etas_mb)
#         ax.plot(vMins_indexed* lightSpeed_kmpers,etas_mb *1e5,color='red',label='Simple Halo Model')
#     ax.legend()
#     ax.set_ylabel('η (s/km)')
#     ax.set_xlabel('Vmin (km/s)')
#     ax.set_xlim([0, 1000])

#     # ax.set_xlim([0, 1000])
#     if save:
#         if plotname is not None:
#             figname = plotname
#         else:
#             figname = dir+'eta_plot.png'
#         plt.savefig(figname, bbox_inches='tight')
#     return #etas,etas_mb,vMins

def plot_eta(mX,params,plot_mb =True,device='cpu',save=False,plotname=None,dir='./',cross_section = 1e-36,halo_params=None,norm=1e-10):
    #halo_params = [238.0,250.2,544,0.3e9] <- acceptable format to input halo params v0,vE,vEsc,rhoX
    import matplotlib.pyplot as plt
    import numpy as np
    from QEDark3 import QEDark
    from QEDarkConstants import ccms,lightSpeed_kmpers,lightSpeed,sec2yr
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

    return 
       
def plot_eta_params(data,name,test_mX,FDMn,plot_shm = False,cross_section=1e-36,halo_params=None):
    import numpy as np
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
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from haloidp_analysis import plot_eta
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
    import pickle
    import os
    
    print(f'looking at files from {mock_data_dir}')
    fpath_si = f'./halo_independent/{mock_data_dir}/Model{modelnumber}_Si.csv'

    from numpy import loadtxt as loadtxt
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
    from tqdm.autonotebook import tqdm
    import pickle
    import os
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
    import pickle
    model = 4
    fpath_si = f'./halo_independent/mock_data/Model{model}_Si.csv'
    from numpy import loadtxt as loadtxt
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