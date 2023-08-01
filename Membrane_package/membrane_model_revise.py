#%% Library import
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import Membrane_pack
from scipy.interpolate import interp1d

from scipy.optimize import minimize
from scipy import interpolate

parameters = {'axes.labelsize': 17,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
          'axes.titlesize': 20}
plt.rcParams.update(parameters)

import pickle
from itertools import product
import warnings

# Constants
Rgas = 8.314*1e9                # mm3 Pa/K mol

######################
#### ODE function ####
######################
def gaode(dy_fun, y0, t, args= None):
#    if np.isscalar(t):
#        t_domain = np.linspace(0,t, 10001, dtype=np.float64)
#    else:
#        t_domain = np.array(t[:], dtype = np.float64)
    t_domain = np.array(t[:], dtype = np.float64)
    y_res = []
    dt_arr = t_domain[1:] - t_domain[:-1]

    N = len(y0)
    tt_prev = t_domain[0]
    y_tmp = np.array(y0, dtype = np.float64)
    y_res.append(y_tmp)
    if args == None:
        for tt, dtt in zip(t_domain[:-1], dt_arr):
            dy_tmp = np.array(dy_fun(y_tmp, tt))
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
#            if tt%10 == 1:
#                print(y_tmp_new, y_tmp)
        y_res_arr = np.array(y_res, dtype = np.float64)
    else:
        for tt, dtt in zip(t_domain[1:], dt_arr):
            dy_tmp = np.array(dy_fun(y_tmp, tt))
            y_tmp_new = y_tmp + dy_tmp*dtt
            tt_prev = tt
            y_res.append(y_tmp_new)
            y_tmp = y_tmp_new
        y_res_arr = np.array(y_res, dtype=object)
    
    return y_res_arr

#%%
class Membrane:
    #### Module sizing parameter ####
    def __init__(self, L, D_inner, D_outer, D_module, N_fiber, n_component, 
                 N_node = 10,):
        
        thickness = (D_outer-D_inner)/2
        Ac_shell = np.pi*D_module**2/4 - N_fiber*np.pi*D_inner**2/4     # (mm^2)              
        self._L = L
        self._D_inner = D_inner
        self._D_outer = D_outer
        self._D_module = D_module
        self._N_fiber = N_fiber
        
        self._Ac_shell = Ac_shell
        self._thickness = thickness
        
        self._n_comp = n_component
        self._N = int(N_node+1)
        self._z = np.linspace(0, self._L, self._N)
        self._required = {'Design':True,
        'membrane_info':False,
        'gas_prop_info': False,
        'mass_trans_info': False,}
        self._required['boundaryC_info'] = False
        self._required['initialC_info'] = False
        
    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n'
        return str_return
    
    #### Membrane property information ####
    def membrane_info(self, a_perm):
        if len(a_perm) != self._n_comp:
            print('Output should be a list/narray including {} narray!'.format(self._n_comp))
        else:
            self._a_perm = a_perm
            self._required['membrane_info'] = True
    
    #### Gas property information ####
    def gas_prop_info(self, Mass_molar, mu_viscosity, rho_density,):
        stack_true = 0
        if len(Mass_molar) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if len(mu_viscosity) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))
        if len(rho_density) == self._n_comp:
            stack_true = stack_true + 1
        else:
            print('The input variable should be a list/narray with shape ({0:d}, ).'.format(self._n_comp))    
        if stack_true == 3:
            self._M_m = Mass_molar
            self._mu = mu_viscosity
            self._rho = rho_density
            self._required['gas_prop_info'] = True
    
    #### Mass transfer coefficient information ####
    def mass_trans_info(self, k_mass_transfer):
        self._k_mtc = k_mass_transfer
        self._required['mass_trans_info'] = True
    
    #### Boundary condition ####
    def boundaryC_info(self, Pf_inlet, T_inlet, y_inlet, F_feed):
        try:
            if len(y_inlet) == self._n_comp:
                self._Pf_in = Pf_inlet
                self._T_in = T_inlet
                self._y_in = y_inlet
                self._F_in = F_feed
                self._required['boundaryC_info'] = True
                
            else:
                print('The inlet composition should be a list/narray with shape (2, ).')
        except:
            print('The inlet composition should be a list/narray with shape (2, ).')
    
    #### co-current model ####
    def CoFs(self, y, z):
        T = self._T_in
        N_f = self._N_fiber
        D_md = self._D_module
        D_i = self._D_inner
        D_o = self._D_outer
        Ac_shell = self._Ac_shell
        a_i = self._a_perm
        rho_i = self._rho
        mu_i = self._mu
        
        F_f = np.array([y[0], y[1]])
        F_p = np.array([y[2], y[3]])
        Pf, Pp =  y[4], y[5]

        F_f_tot = np.sum(F_f, axis=0)           
        F_p_tot = np.sum(F_p, axis=0)

        x_i = F_f/F_f_tot
        y_i = F_p/F_p_tot
        
        Pf_i = x_i*Pf
        Pp_i = y_i*Pp
        
        mu_f = np.sum(mu_i*x_i)     # feed side viscosity (Pa s)
        mu_p = np.sum(mu_i*y_i)     # permeate side viscosity (Pa s)
        rho_mix_f = rho_i*y_i       # Density (kg/mm3)

        dPfdz = - 192*N_f*D_o*(D_md + N_f*D_o)*Rgas*T*mu_f*F_f_tot/( np.pi*(D_md*D_md - N_f*D_o*D_o)**3 * Pf )*1E-10
        dPpdz = - 128*Rgas*T*mu_p*F_p_tot/( np.pi*D_i**4 * N_f * Pp )*1E-10

        J = a_i*(Pf_i - Pp_i)
        arg_neg_J = J < 0
        J[arg_neg_J] = 0
 
        dF_f = -D_o*np.pi*J*N_f
        dF_p = D_o*np.pi*J*N_f
        
        dF_f = dF_f.tolist()
        dF_p = dF_p.tolist()

        dydz = dF_f+ dF_p+ [dPfdz]+[dPpdz]
        
        return dydz
    
    #### Estimation of permeate side pressure in z=0 ####
    def find_Pp_in(self, y):
        Pp_list = np.linspace(1.000000001, 1.2, 50)
        err_list = []
        Pp_out_list = []
        for Pp_in in Pp_list:
            y[-1] = Pp_in
            y_res = gaode(self.CoFs, y, self._z,)
                    
            Pp_out = y_res[-1,5]
            Pp_out_list.append(Pp_out)

        P_reduced = np.array(Pp_out_list) - 1
        func = interpolate.UnivariateSpline(Pp_list, P_reduced, s=0)
        Pp_sol_list = func.roots()
        
        err_list = []
        if len(Pp_sol_list) > 1:
            for Pp_new in Pp_sol_list:
                y[-1] = Pp_new
                y_res = gaode(self.CoFs, y, self._z,)
                
                Pp_out = y_res[-1,5]
                err_list.append((Pp_out-1)**2)
            Pp_sol = Pp_sol_list[np.argmin(np.array(err_list))]
        elif len(Pp_sol_list)==1:
            Pp_sol = Pp_sol_list[0]
        else:
            Pp_sol = Pp_list[0]
        return Pp_sol
   
    #### Initial condition ####
    def initialC_info(self, configuration):
        self._config = configuration
        try:
            if self._config == 'co':
                Fp_H2_in = 1e-6     # initial value
                Fp_N2_in = 1e-6     # initial value
                Fp_init = np.array([Fp_H2_in,Fp_N2_in])
                Pp_init = 1.01
                if len(Fp_init) == self._n_comp:
                    self._Pp_in = Pp_init
                    self._Fp_in = Fp_init
                    def Fp_initial(x):
                        a_perm = self._a_perm
                        y_feed = self._y_in
                        Pf_z0 = self._Pf_in
                        Pp_z0 = self._Pp_in
                        D_outer = self._D_outer
                        N_fiber = self._N_fiber
                        L = self._L
                            
                        fp_H2, fp_N2 = x[0], x[1]
                        
                        penalty = 0
                        if fp_H2 < 10**-6:
                            penalty += np.sqrt((fp_H2-10**-6)**2)
                            fp_H2 = 10**-6
                        if fp_N2 < 10**-6:
                            penalty += np.sqrt((fp_N2-10**-6)**2)
                            fp_N2 = 10**-6
                        
                        fp = fp_H2+fp_N2
                        yH2, yN2 = fp_H2/fp, fp_N2/fp
                        J_H2 = a_perm[0]*(Pf_z0*y_feed[0]-Pp_z0*yH2)
                        J_N2 = a_perm[1]*(Pf_z0*y_feed[1]-Pp_z0*yN2)

                        f_H2_cal = D_outer*np.pi*J_H2*N_fiber*L/N_fiber
                        f_N2_cal = D_outer*np.pi*J_N2*N_fiber*L/N_fiber

                        err_H2 = (fp_H2-f_H2_cal)**2
                        err_N2 = (fp_N2-f_N2_cal)**2
                        return err_H2+err_N2+penalty
                    
                    res = minimize(Fp_initial, self._Fp_in, 
                                    method='SLSQP',          #initial guess of permeate flowrate
                                    options={'ftol':1e-15,'disp':False})
                    Fp_z0 = list(res.x)
                    Ff_z0= [self._F_in[i]-Fp_z0[i] for i in range(len(Fp_z0))]
                    y0 = np.array(Ff_z0 + Fp_z0 + [self._Pf_in, self._Pp_in])               
                    Pp_z0 = self.find_Pp_in(y0)
                    y0[-1] = Pp_z0
                    
                    self._y0 = y0
                    self._required['initialC_info'] = True
                    
                else:
                    print('The inlet composition should be a list/narray with shape (2, ).')

            elif self._config == 'ct':
                self.Fp_H2_in = self._F_in[0]*0.5     # initial guess
                self.Fp_N2_in = self._F_in[1]*0.05     # initial guess
                Fp_init = np.array([self.Fp_H2_in,self.Fp_N2_in])
                self._Pp_in = 1
                if len(Fp_init) == self._n_comp:
                    y0 = np.array(self._F_in + [self.Fp_H2_in, self.Fp_N2_in] + [self._Pf_in, self._Pp_in])
                    self._y0 = y0
                    self._required['initialC_info'] = True
                else:
                    print('The inlet composition should be a list/narray with shape (2, ).')

        except:
            print('Except: The inlet composition should be a list/narray with shape (2, ).')
    
    #### Simulation run ####
    def run_mem(self, n_stage, mode, P_down = False):
        self._n_stage = n_stage
        self._mode = mode
        
        if self._config == 'co':
            _memfunc = self.CoFs
            y_res = gaode(self.CoFs, self._y0, self._z,)
            if self._n_stage == 1:
                self._y_res = y_res
        
            else:
                if self._mode == 'P2F':
                    res_list = []
                    res_list.append(y_res)
                    for ii in range(self._n_stage-1):
                        self._Pf_in = P_down[ii]
                        self._F_in = y_res[-1, self._n_comp:self._n_comp*2]
                        _y = self._F_in / np.sum(self._F_in)
                        self.boundaryC_info(self._Pf_in, self._T_in, _y, self._F_in)
                        self.initialC_info(self._config)
                        y_res = gaode(self.CoFs, self._y0, self._z,)
                        res_list.append(y_res)
                    self._y_res = res_list
                    
                elif self._mode == 'R2F':
                    res_list = []
                    res_list.append(y_res)
                    for ii in range(self._n_stage-1):
                        self._Pf_in = P_down[ii]
                        self._F_in = y_res[-1, :self._n_comp]
                        _y = self._F_in / np.sum(self._F_in)
                        self.boundaryC_info(self._Pf_in, self._T_in, _y, self._F_in)
                        self.initialC_info(self._config)
                        y_res = gaode(self.CoFs, self._y0, self._z,)
                        res_list.append(y_res)
                    self._y_res = res_list    
                else:
                    print('Invalide mode!: Type P2F or R2F')
                    
                    
        if self._config == 'ct':

            tol = 1e-7
            for ii in range(20000):
                y_res = gaode(self.CtFs, self._y0, self._z,)
                
                self.Fp_H2_in = y_res[0,2]
                self.Fp_N2_in = y_res[0,3]
                F_f = np.array([y_res[:,0], y_res[:,1]])
                F_p = np.array([y_res[:,2], y_res[:,3]])
                
                Pf = y_res[:,4]
                Pp = y_res[:,5]

                x_i = F_f/np.sum(F_f, axis=0)
                y_i = F_p/np.sum(F_p, axis=0)
                
                J = (self._a_perm).reshape(-1,1)*(x_i*Pf - y_i*Pp)#*1E5
                
                arg_neg_J = J < 0
                J[arg_neg_J] = 0
                
                # arg_neg_f = F_f < 1e-20
                # arg_neg_p = F_p < 1e-20
                
                # J[arg_neg_f] = 0
                # J[arg_neg_p] = 0
                
                #Error calculation  
                err1 = (self.Fp_H2_in-(np.pi*self._D_outer*self._L/self._N*self._N_fiber)*sum(J[0,:]))/self.Fp_H2_in
                err2 = (self.Fp_N2_in-(np.pi*self._D_outer*self._L/self._N*self._N_fiber)*sum(J[1,:]))/self.Fp_N2_in

                tol = abs(err1)+abs(err2)
                
                Kg = 0.1
                self.Fp_H2_in = self.Fp_H2_in - Kg*err1*self.Fp_H2_in
                self.Fp_N2_in = self.Fp_N2_in - Kg*err2*self.Fp_N2_in
                self._y0 = np.array(self._F_in+ [self.Fp_H2_in, self.Fp_N2_in]+ [self._Pf_in, self._Pp_in])
                
                print('Cycle {0:03d}: Err = {1:.10e} // Fp_H2 = {2:.4e} mol/s'.format(ii, tol, self.Fp_H2_in))
                if abs(tol) < 1E-7:
                    break            

            if self._n_stage == 1:
                self._y_res = y_res
        
            else:
                if self._mode == 'P2F':
                    res_list = []
                    res_list.append(y_res)
                    for ii in range(self._n_stage-1):
                        self._Pf_in = P_down[ii]
                        self._F_in = y_res[0, self._n_comp:self._n_comp*2]
                        _y = self._F_in / np.sum(self._F_in)
                        self.boundaryC_info(self._Pf_in, self._T_in, _y, self._F_in)
                        self.initialC_info(self._config)
                        y_res = gaode(self.CtFs, self._y0, self._z,)
                        res_list.append(y_res)
                    self._y_res = res_list
                    
                elif self._mode == 'R2F':
                    res_list = []
                    res_list.append(y_res)
                    for ii in range(self._n_stage-1):
                        self._Pf_in = P_down[ii]
                        self._F_in = y_res[0, :self._n_comp]
                        _y = self._F_in / np.sum(self._F_in)
                        self.boundaryC_info(self._Pf_in, self._T_in, _y, self._F_in)
                        self.initialC_info(self._config)
                        y_res = gaode(self.CtFs, self._y0, self._z,)
                        res_list.append(y_res)
                    self._y_res = res_list    
                else:
                    print('Invalide mode!: Type P2F or R2F')
        return self._y_res
    
    def MassBalance(self,):
        err_list = []
        if self._n_stage == 1:
            y = self._y_res
            inpt = sum(y[0,:4])
            outp = sum(y[-1,:4])
            
            err = abs(inpt-outp)/inpt*100
            print('Mass balance (error %): ', err)
            return err
        else:
            for y in self._y_res:
                inpt = sum(y[0,:4])
                outp = sum(y[-1,:4])
                
                err = abs(inpt-outp)/inpt*100
                print('Mass balance (error %): ', err)
                err_list.append(err)
            return err_list
    
    def Comp_cost(self, h_capa_ratio, comp_eiff):
        P_ref = 1                    # inlet pressure (1bar)
        total_CR = self._Pf_in/1   # total compression ratio
        R_gas = 8.314                # Gas constant (J/K mol)
        NN = int(np.log(total_CR)/np.log(2.5))+1     # # of compressors
        
        cost = 0
        work = 0
        for i in range(NN):
            
            effi = comp_eiff-(i*0.05)
            if i != NN-1:
                work_tmp = np.sum(self._F_in)*R_gas*self._T_in/effi*(h_capa_ratio/(h_capa_ratio-1))*((2.5*P_ref/P_ref)**((h_capa_ratio-1)/h_capa_ratio)-1)
            else:
                work_tmp = np.sum(self._F_in)*R_gas*self._T_in/effi*(h_capa_ratio/(h_capa_ratio-1))*((self._Pf_in/P_ref)**((h_capa_ratio-1)/h_capa_ratio)-1)
            
            work += work_tmp
            cost_tmp = 5840 * (work_tmp * 0.001)**0.82
            cost += cost_tmp
            P_ref = 2.5*P_ref
        return work*0.001, cost  


    def GRC(self, unit_m_cost, mem_life, yr, interest_rate, h_capa_ratio, comp_eiff):
        
        area = np.pi*self._D_outer*self._L*self._N_fiber     # membrane area (mm2)
        total_feed= np.sum(self._F_in)*self._N_fiber
        c_work, c_cost = self.Comp_cost(h_capa_ratio, comp_eiff,)
        mem_cost = unit_m_cost * area
        
        capex = c_cost+mem_cost
        TCI = 1.4*capex     # sum up working capital, fixed capital etc
        
        AF = (1-(1/(1+interest_rate)**yr))/interest_rate      # Annualized factor
        EAC = TCI / AF              # Annualized capital cost
        # TFI = 1.344*(area*MODP+c_cost)
        
        # opex
        FC = capex*0.014
        MRC = (mem_cost/2)/mem_life        # membrane replacement cost
        elect_cost = c_work*0.071    # 전기요금
        M = capex * 0.01              # Maintenance
        
        TPC = (FC + MRC + M*1.6 + elect_cost)/(1-0.26)    # Total product cost
        TAC = EAC + TPC             # Total annalized cost

        return TAC
    
        #### Profile plot ####
    def results_plot_kci_co(self, z_ran=False):
        z_dom = self._z
        a_perm = self._a_perm
        if self._n_stage == 1:
            y_list = [self._y_res]
        else:
            y_list = self._y_res
            
        for y_plot in y_list:
            Ff_H2,Ff_N2, Fp_H2, Fp_N2, Pf, Pp = y_plot[:,0], y_plot[:,1], y_plot[:,2], y_plot[:,3], y_plot[:,4], y_plot[:,5] 
            x_H2 = Ff_H2/(Ff_H2+Ff_N2)
            x_N2 = Ff_N2/(Ff_H2+Ff_N2)
            y_H2 = Fp_H2/(Fp_H2+Fp_N2)
            y_N2 = Fp_N2/(Fp_H2+Fp_N2)
            
            x_i = np.array([x_H2, x_N2])
            y_i = np.array([y_H2, y_N2])
            J =(x_i*Pf - y_i*Pp) * a_perm.reshape(-1, 1)
            arg_neg_J = J < 0
            J[arg_neg_J] = 0
            
            ########### flux  ##########
            fig = plt.figure(figsize=(10,7),dpi=90)
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            ax1 = fig.add_subplot(221)
            ax1.plot((z_dom[1:]*1e-3), (J[0][1:]*1e6), linewidth=2,color = 'b', label='$J_{H_2}$')
            ax1.plot((z_dom[1:]*1e-3), (J[1][1:]*1e6), linewidth=2,color = 'r', label='$J_{N_2}$')
            ax1.set_xlabel('z (m)')
            ax1.set_ylabel('fluxes [mol/(m2 s)]')
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax1.legend(fontsize=13, loc='best')
            # plt.xlim([0, z_dom[-1]*1e-3])
            if z_ran:
                plt.xlim(z_ran)
            ax1.grid(linestyle='--')
            
            ########### Flowrate  ##########
            ax2 = fig.add_subplot(222)
            ax2.plot(z_dom[1:]*1e-3, Ff_H2[1:], linewidth=2,color = 'b', label='$Feed_{H_2}$')
            ax2.plot(z_dom[1:]*1e-3, Ff_N2[1:], linewidth=2,color = 'b', linestyle='--', label='$Feed{N_2}$')
            ax2.set_xlabel('z (m)')
            ax2.set_ylabel('feed flowrate (mol/s)')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax2.grid(linestyle='--')
            ax3 = ax2.twinx()
            ax3.plot(z_dom[1:]*1e-3, Fp_H2[1:], linewidth=2, color = 'r', label='$Perm_{H_2}$', )
            ax3.plot(z_dom[1:]*1e-3, Fp_N2[1:], linewidth=2, color = 'r', linestyle='--', label='$Perm_{N_2}$')
            ax3.set_ylabel('Permeate flowrate (mol/s)')
            ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax2.yaxis.label.set_color('b')
            ax3.yaxis.label.set_color('r')
            
            ax3.spines["right"].set_edgecolor('r')
            ax3.spines["left"].set_edgecolor('b')
            if z_ran:
                plt.xlim(z_ran)
            ax2.tick_params(axis='y', colors='b')
            ax3.tick_params(axis='y', colors='r')
            # plt.xlim([0,0.1])

            ########### Mole fraction ##########
            ax4 = fig.add_subplot(223)
            ax4.plot((z_dom[1:]*1e-3), (x_H2[1:]), linewidth=2, color='b', label='$x_{H_2}$')
            ax4.plot((z_dom[1:]*1e-3), (x_N2[1:]), linewidth=2, color='b', label='$x_{N_2}$', linestyle='--')
            ax4.set_xlabel('z (m)')
            plt.ylim([0, 1]) 
            ax4.set_ylabel('mole fraction (mol/mol)')
            ax4.grid(linestyle='--')
            ax5 = ax4.twinx()
            ax5.plot((z_dom[1:]*1e-3), (y_H2[1:]), linewidth=2, color='r', label='$y_{H_2}$')
            ax5.plot((z_dom[1:]*1e-3), (y_N2[1:]), linewidth=2, color='r', label='$y_{N_2}$', linestyle='--')
            plt.ylim([-0.01, 1.01])    
            if z_ran:
                plt.xlim(z_ran)
            ax4.yaxis.label.set_color('b')
            ax5.yaxis.label.set_color('r')

            ax4.tick_params(axis='y', colors='b')
            ax5.tick_params(axis='y', colors='r')
            
            ax5.spines["right"].set_edgecolor('r')
            ax5.spines["left"].set_edgecolor('b')
            
            ########### Pressure drop ##########
            ax6 = fig.add_subplot(224)
            ax6.plot(z_dom*1e-3, (Pf[0]-Pf)*1e5, 'b-', label = 'Feed side')
            ax6.set_xlabel('z (m)')
            ax6.set_ylabel('$\\vartriangle$ $P_{f}$ (Pa)')
            ax6.ticklabel_format(axis='y', style='plain')
            ax6.grid(linestyle='--')
            ax7= ax6.twinx()
            ax7.plot(z_dom*1e-3, (Pp[0]-Pp)*1e5, 'r-', label = 'Permeate side')
            ax7.set_ylabel('$\\vartriangle$ $P_{p}$ (Pa)')
            fig.tight_layout()
            ax6.yaxis.label.set_color('b')
            ax7.yaxis.label.set_color('r')

            ax6.tick_params(axis='y', colors='b')
            ax7.tick_params(axis='y', colors='r')
            
            ax7.spines["right"].set_edgecolor('r')
            ax7.spines["left"].set_edgecolor('b')
            # plt.xlim([0, 0.005])
            if z_ran:
                plt.xlim(z_ran)
            
            plt.show()
            
    def results_plot_kci_ct(self, z_ran=False):
        z_dom = self._z
        a_perm = self._a_perm
        if self._n_stage == 1:
            y_list = [self._y_res]
        else:
            y_list = self._y_res
            
        for y_plot in y_list:
            Ff_H2,Ff_N2, Fp_H2, Fp_N2, Pf, Pp = y_plot[:,0], y_plot[:,1], y_plot[:,2], y_plot[:,3], y_plot[:,4], y_plot[:,5] 
            x_H2 = Ff_H2/(Ff_H2+Ff_N2)
            x_N2 = Ff_N2/(Ff_H2+Ff_N2)
            y_H2 = Fp_H2/(Fp_H2+Fp_N2)
            y_N2 = Fp_N2/(Fp_H2+Fp_N2)
            
            x_i = np.array([x_H2, x_N2])
            y_i = np.array([y_H2, y_N2])
            J =(x_i*Pf - y_i*Pp) * a_perm.reshape(-1, 1)
            arg_neg_J = J < 0
            J[arg_neg_J] = 0
            
            ########### flux  ##########
            fig = plt.figure(figsize=(10,7),dpi=90)
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            ax1 = fig.add_subplot(221)
            ax1.plot((z_dom*1e-3), (J[0]*1e6), linewidth=2,color = 'b', label='$J_{H_2}$')
            ax1.plot((z_dom*1e-3), (J[1]*1e6), linewidth=2,color = 'r', label='$J_{N_2}$')
            ax1.set_xlabel('z (m)')
            ax1.set_ylabel('fluxes [mol/(m2 s)]')
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax1.legend(fontsize=13, loc='best')
            # plt.xlim([0, z_dom[-1]*1e-3])
            if z_ran:
                plt.xlim(z_ran)
            ax1.grid(linestyle='--')
            
            ########### Flowrate  ##########
            ax2 = fig.add_subplot(222)
            ax2.plot(z_dom*1e-3, Ff_H2, linewidth=2,color = 'b', label='$Feed_{H_2}$')
            ax2.plot(z_dom*1e-3, Ff_N2, linewidth=2,color = 'b', linestyle='--', label='$Feed{N_2}$')
            ax2.set_xlabel('z (m)')
            ax2.set_ylabel('feed flowrate (mol/s)')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax2.grid(linestyle='--')
            ax3 = ax2.twinx()
            ax3.plot(z_dom*1e-3, Fp_H2, linewidth=2, color = 'r', label='$Perm_{H_2}$', )
            ax3.plot(z_dom*1e-3, Fp_N2, linewidth=2, color = 'r', linestyle='--', label='$Perm_{N_2}$')
            ax3.set_ylabel('Permeate flowrate (mol/s)')
            ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax2.yaxis.label.set_color('b')
            ax3.yaxis.label.set_color('r')
            
            ax3.spines["right"].set_edgecolor('r')
            ax3.spines["left"].set_edgecolor('b')
            if z_ran:
                plt.xlim(z_ran)
            ax2.tick_params(axis='y', colors='b')
            ax3.tick_params(axis='y', colors='r')
            # plt.xlim([0,0.1])

            ########### Mole fraction ##########
            ax4 = fig.add_subplot(223)
            ax4.plot((z_dom*1e-3), (x_H2), linewidth=2, color='b', label='$x_{H_2}$')
            ax4.plot((z_dom*1e-3), (x_N2), linewidth=2, color='b', label='$x_{N_2}$', linestyle='--')
            ax4.set_xlabel('z (m)')
            plt.ylim([0, 1]) 
            ax4.set_ylabel('mole fraction (mol/mol)')
            ax4.grid(linestyle='--')
            ax5 = ax4.twinx()
            ax5.plot((z_dom*1e-3), (y_H2), linewidth=2, color='r', label='$y_{H_2}$')
            ax5.plot((z_dom*1e-3), (y_N2), linewidth=2, color='r', label='$y_{N_2}$', linestyle='--')
            plt.ylim([-0.01, 1.01])    
            if z_ran:
                plt.xlim(z_ran)
            ax4.yaxis.label.set_color('b')
            ax5.yaxis.label.set_color('r')

            ax4.tick_params(axis='y', colors='b')
            ax5.tick_params(axis='y', colors='r')
            
            ax5.spines["right"].set_edgecolor('r')
            ax5.spines["left"].set_edgecolor('b')
            
            ########### Pressure drop ##########
            ax6 = fig.add_subplot(224)
            ax6.plot(z_dom*1e-3, (Pf[0]-Pf)*1e5, 'b-', label = 'Feed side')
            ax6.set_xlabel('z (m)')
            ax6.set_ylabel('$\\vartriangle$ $P_{f}$ (Pa)')
            ax6.ticklabel_format(axis='y', style='plain')
            ax6.grid(linestyle='--')
            ax7= ax6.twinx()
            ax7.plot(z_dom*1e-3, (Pp[-1]-Pp)*1e5, 'r-', label = 'Permeate side')
            ax7.set_ylabel('$\\vartriangle$ $P_{p}$ (Pa)')
            fig.tight_layout()
            ax6.yaxis.label.set_color('b')
            ax7.yaxis.label.set_color('r')

            ax6.tick_params(axis='y', colors='b')
            ax7.tick_params(axis='y', colors='r')
            
            ax7.spines["right"].set_edgecolor('r')
            ax7.spines["left"].set_edgecolor('b')
            # plt.xlim([0, 0.005])
            if z_ran:
                plt.xlim(z_ran)
            
            plt.show()
            
                         
# #%%
# #%%
# # Sizing parameters
# D_inner = 200*1e-3            # Membrane inner diameter (mm)
# D_outer = 250*1e-3            # Membrane outer diameter (mm)
# D_module = 0.1*1e3            # Module diameter (mm)
# N_fiber = 60000               # number of fiber (-)
# L = 0.6*1e3                   # fiber length (mm)
# n_component = 2

# mem = Membrane(L, D_inner, D_outer, D_module, N_fiber, n_component, N_node = 1e3)
# print(mem)
# # %%
# a_perm = np.array([3.207e-9, 1.33e-10])*1e-6*1e5 #Permeance(mol/(mm2 bar s))
# mem.membrane_info(a_perm)
# print(mem)
# # %%
# Mw_i = np.array([44e-3, 16e-3])     # Molar weight (kg/mol)
# rho_i = np.array([1.98, 0.657])*1e-9     # Density (kg/mm3)
# mu_H2 = 0.0155e-3           # H2 viscosity (Pa s)
# mu_N2 = 0.011e-3           # N2 viscosity (Pa s)
# # viscosity values from https://www.engineeringtoolbox.com/gases-absolute-dynamic-viscosity-d_1888.html
# mu_i = np.array([mu_H2, mu_N2])

# mem.gas_prop_info(Mw_i, mu_i, rho_i)
# print(mem)
# # %%
# k_mass = 1e-1               # Mass transfer coeff. (mm/s)
# mem.mass_trans_info(k_mass)
# print(mem)
# # %%
# # Operating conditions
# P_feed = 60                # pressure of feed side (bar)
# T = 296.15
# y_feed = np.array([0.1, 0.9])     # mole fraction (CO2, CH4)
# F_feed = 0.175
# Ff_z0_init = list(y_feed*F_feed)

# mem.boundaryC_info(P_feed, T, y_feed, Ff_z0_init)
# print(mem)
# # %%
# Fp_H2_in = 1e-6     # initial value
# Fp_N2_in = 1e-6     # initial value
# Fp_init = np.array([Fp_H2_in,Fp_N2_in])
# Pp_z0 = 1.01      # initial guess

# mem.initialC_info(Fp_init, Pp_z0)
# print(mem)
# # %%
# res = mem.run_mem(2, 'P2F')
# # %%

# err = mem.MassBalance()
# # %%
# mem.results_plot_kci()
# # %%

# %%
