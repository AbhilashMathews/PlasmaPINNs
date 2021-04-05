# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 09:14:10 2021

@author: mathewsa
"""

import h5py
import numpy as np

save_directory = '/home/mathewsa/Plasma-PINN_manuscript_v0/no_noise/' #path to find files and save them

N_outputs = 1 #number of predictions made by each PINN
timelen_end = 20.0 #training time (hours)

layers = [3, 50, 50, 50, 50, 50, N_outputs]

#.Size of data set
nx0G = 256
ny0G = 128
nz0G = 32
#.Domain size
a0 = 0.22             #. minor radius (m)
R0 = 0.68 + a0        #. major radius (m)
Lx = 0.35             #. Radial size of simulation domain (normalized by a0)
Ly = 0.25             #. Vertical size of simulation domain (normalized by a0)
Lz = 20.0             #. Connection length (normalized by R0)

dt        = 0.000005  #.time step. (normalized)
nts       = 16000     #.number of time steps per frame.
preFrames = 0         #.time frames in previous job this one restarted from.
iFrame  = [0]         #.First frame to output (zero indexed).
fFrame  = [398]       #.Last frame to output.
t_start = 0.0 

len_frames = fFrame[0] - iFrame[0] + 1
t_end = t_start + nts*dt*(len_frames - 1)
x_t = np.linspace(t_start,t_end,len_frames) #dimensionless time
tmp = Lx
Lx = np.array([tmp, Ly, Lz, t_end])
tmp = nx0G
nx0G = np.array([tmp, ny0G, nz0G, len_frames])

#.Cell spacing
dx = np.array([Lx[0]/nx0G[0], Lx[1]/nx0G[1], Lx[2]/nx0G[2], dt]) 

x = [ (np.arange(0,nx0G[0])-nx0G[0]/2.0)*dx[0]+dx[0]/2.0, \
      (np.arange(0,nx0G[1])-nx0G[1]/2.0)*dx[1]+dx[1]/2.0, \
      (np.arange(0,nx0G[2])-nx0G[2]/2.0)*dx[2]+dx[2]/2.0, \
      x_t]

#normalized diffusion coefficients applied in the code
#note: implicit diffusion applies an additional dt factor
DiffX = 2.*np.pi/(dx[0]*3.)
DiffY = 2.*np.pi/(dx[1]*3.)
DiffZ = 2.*np.pi/(dx[2]*3.)

DiffX_norm = DiffX**2.
DiffY_norm = DiffY**2.
DiffZ_norm = DiffZ**2.

data_file = str(save_directory)+'PlasmaPINN_data_inputs_paper.h5' 
h5f = h5py.File(data_file, "r")
x_x = h5f['x_x'].value 
x_y = h5f['x_y'].value  
x_z = h5f['x_z'].value  
x_t = h5f['x_t'].value   
y_den = h5f['y_den'].value  
y_Te = h5f['y_Te'].value 

init_weight_den = (1./np.median(np.abs(y_den)))
init_weight_Te = (1./np.median(np.abs(y_Te)))

frac_train = 1.0
N_train = int(frac_train*len(y_den))
idx = np.random.choice(len(y_den), N_train, replace=False)

x_train = x_x[idx,:]
y_train = x_y[idx,:]
z_train = x_z[idx,:]
t_train = x_t[idx,:]
v1_train = y_den[idx,:] 
v5_train = y_Te[idx,:] 

sample_batch_size = int(500)

import time
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class PhysicsInformedNN:
    def __init__(self, x, y, t, v1, v5, layers):
        X = np.concatenate([x, y, t], 1) 
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.v1 = v1
        self.v5 = v5
        self.layers = layers
        self.current_time = 0.0
        self.start_time = 0.0 
        self.weights_v1, self.biases_v1 = self.initialize_NN(layers)
        self.weights_v2, self.biases_v2 = self.initialize_NN(layers)
        self.weights_v3, self.biases_v3 = self.initialize_NN(layers)
        self.weights_v4, self.biases_v4 = self.initialize_NN(layers)
        self.weights_v5, self.biases_v5 = self.initialize_NN(layers) 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                                     log_device_placement=False,
                                                     device_count={ "CPU": 32},
                                                     inter_op_parallelism_threads=1,
                                                     intra_op_parallelism_threads=32)) 
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.v1_tf = tf.placeholder(tf.float32, shape=[None, self.v1.shape[1]])
        self.v5_tf = tf.placeholder(tf.float32, shape=[None, self.v5.shape[1]]) 
        self.v1_pred, self.v5_pred,\
        self.PINN_v2_pred, self.PINN_v3_pred, self.PINN_v4_pred,\
        self.f_v1_pred, self.f_v5_pred = self.net_plasma(self.x_tf, self.y_tf, self.t_tf) 
        self.loss1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.v1_tf - self.v1_pred))
        self.loss5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(self.v5_tf - self.v5_pred))
        self.lossf1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.f_v1_pred))
        self.lossf5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(self.f_v5_pred))
        self.optimizer_v1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_v5 = tf.contrib.opt.ScipyOptimizerInterface(self.loss5,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v5+self.biases_v5,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_f = tf.contrib.opt.ScipyOptimizerInterface((self.lossf1 + self.lossf5),
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer() 
        self.loss_v1_log = []
        self.loss_v5_log = []
        self.loss_f1_log = []
        self.loss_f5_log = []
        self.train_op_Adam_v1 = self.optimizer_Adam.minimize(self.loss1, var_list=self.weights_v1+self.biases_v1)
        self.train_op_Adam_v5 = self.optimizer_Adam.minimize(self.loss5, var_list=self.weights_v5+self.biases_v5)
        self.train_op_Adam_f = self.optimizer_Adam.minimize(self.lossf1 + self.lossf5, var_list=self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b) 
        return weights, biases
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim)) 
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        X_d0 = max(self.ub[0] - self.lb[0], 1e-6)
        X_d1 = max(self.ub[1] - self.lb[1], 1e-6)
        X_d2 = max(self.ub[2] - self.lb[2], 1e-6)
        X_d = np.array([X_d0, X_d1, X_d2])
        H = 2.0*(X - self.lb)/X_d - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) 
        return Y
    def net_plasma(self, x, y, t): 
        mi_me = 3672.3036
        eta = 63.6094
        nSrcA = 20.0
        enerSrceA = 0.001 
        xSrc = -0.15
        sigSrc = 0.01
        B = (0.22+0.68)/(0.68 + 0.22 + a0*x)
        eps_R = 0.4889
        eps_v = 0.3496
        alpha_d = 0.0012
        tau_T = 1.0
        kappa_e = 7.6771
        kappa_i = 0.2184
        eps_G = 0.0550
        eps_Ge = 0.0005 
        v1 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v1, self.biases_v1)
        v2 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v2, self.biases_v2)
        v3 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v3, self.biases_v3)
        v4 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v4, self.biases_v4)
        v5 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v5, self.biases_v5)
        PINN_v2 = v2
        PINN_v3 = v3
        PINN_v4 = v4 
        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v2_x = tf.gradients(PINN_v2, x)[0]
        v2_y = tf.gradients(PINN_v2, y)[0]
        v5_t = tf.gradients(v5, t)[0]
        v5_x = tf.gradients(v5, x)[0]
        v5_y = tf.gradients(v5, y)[0] 
        pe = v1*v5
        pe_y = tf.gradients(pe, y)[0]
        jp = v1*((tau_T**0.5)*v4 - v3) 
        lnn = tf.log(v1)
        lnn_x = tf.gradients(lnn, x)[0]
        lnn_xx = tf.gradients(lnn_x, x)[0]
        lnn_xxx = tf.gradients(lnn_xx, x)[0]
        lnn_xxxx = tf.gradients(lnn_xxx, x)[0] 
        lnn_y = tf.gradients(lnn, y)[0]
        lnn_yy = tf.gradients(lnn_y, y)[0]
        lnn_yyy = tf.gradients(lnn_yy, y)[0]
        lnn_yyyy = tf.gradients(lnn_yyy, y)[0] 
        Dx_lnn = -((50./DiffX_norm)**2.)*lnn_xxxx
        Dy_lnn = -((50./DiffY_norm)**2.)*lnn_yyyy
        D_lnn = (Dx_lnn + Dy_lnn) 
        lnTe = tf.log(v5)
        lnTe_x = tf.gradients(lnTe, x)[0]
        lnTe_xx = tf.gradients(lnTe_x, x)[0]
        lnTe_xxx = tf.gradients(lnTe_xx, x)[0]
        lnTe_xxxx = tf.gradients(lnTe_xxx, x)[0]
        lnTe_y = tf.gradients(lnTe, y)[0]
        lnTe_yy = tf.gradients(lnTe_y, y)[0]
        lnTe_yyy = tf.gradients(lnTe_yy, y)[0]
        lnTe_yyyy = tf.gradients(lnTe_yyy, y)[0]
        Dx_lnTe = -((50./DiffX_norm)**2.)*lnTe_xxxx
        Dy_lnTe = -((50./DiffY_norm)**2.)*lnTe_yyyy
        D_lnTe = (Dx_lnTe + Dy_lnTe) 
        S_n = nSrcA*tf.exp(-(x - xSrc)*(x - xSrc)/(2.*sigSrc*sigSrc))
        S_Ee = enerSrceA*tf.exp(-(x - xSrc)*(x - xSrc)/(2.*sigSrc*sigSrc)) 
        cond1Sn = tf.greater(S_n[:,0], 0.01*tf.ones(sample_batch_size)) 
        S_n = tf.where(cond1Sn, S_n[:,0], 0.001*tf.ones(sample_batch_size))
        cond1SEe = tf.greater(S_Ee[:,0], 0.01*tf.ones(sample_batch_size))
        S_Ee = tf.where(cond1SEe, S_Ee[:,0], 0.001*tf.ones(sample_batch_size))
        cond2Sn = tf.greater(x, xSrc*tf.ones(sample_batch_size))
        S_n = tf.where(cond2Sn[:,0], S_n, 0.5*tf.ones(sample_batch_size))
        cond2SEe = tf.greater(x, xSrc*tf.ones(sample_batch_size))
        S_Ee = tf.where(cond2SEe[:,0], S_Ee, 0.5*tf.ones(sample_batch_size))
        cond4Sn = tf.greater(v1[:,0], 5.0*tf.ones(sample_batch_size))
        S_n = tf.where(cond4Sn, 0.0*tf.ones(sample_batch_size), S_n)
        cond4SEe = tf.greater(v5[:,0], 1.0*tf.ones(sample_batch_size))
        S_Ee = tf.where(cond4SEe, 0.0*tf.ones(sample_batch_size), S_Ee)
        f_v1 = v1_t + (1./B)*(v2_y*v1_x - v2_x*v1_y) - (-eps_R*(v1*v2_y - alpha_d*pe_y) + S_n + v1*D_lnn)
        f_v5 = v5_t + (1./B)*(v2_y*v5_x - v2_x*v5_y) - v5*(5.*eps_R*alpha_d*v5_y/3. +\
                (2./3.)*(-eps_R*(v2_y - alpha_d*pe_y/v1) +\
                (1./v1)*(0.71*eps_v*(0.0) + eta*jp*jp/(v5*mi_me))) +\
                (2./(3.*pe))*(S_Ee) + D_lnTe) 
        return v1, v5, PINN_v2, PINN_v3, PINN_v4, f_v1, f_v5
    def callback(self, loss1, loss5, lossf1, lossf5):
        global Nfeval
        print(str(Nfeval)+' - PDE loss in loop: %.3e, %.3e, %.3e, %.3e' % (loss1, loss5, lossf1, lossf5))
        Nfeval += 1
    def fetch_minibatch(self, x_in, y_in, t_in, den_in, Te_in, N_train_sample):
        idx_batch = np.random.choice(len(x_in), N_train_sample, replace=False)
        x_batch = x_in[idx_batch,:]
        y_batch = y_in[idx_batch,:]
        t_batch = t_in[idx_batch,:]
        v1_batch = den_in[idx_batch,:]
        v5_batch = Te_in[idx_batch,:]
        return x_batch, y_batch, t_batch, v1_batch, v5_batch
    def train(self, timelen_end): 
        self.start_time = time.time()
        self.current_time = time.time() - self.start_time
        try:
            it = 0
            while self.current_time < timelen_end:
                it = it + 1
                print('Full iteration: '+str(it))
                x_res_batch, y_res_batch, t_res_batch, v1_res_batch, v5_res_batch = self.fetch_minibatch(self.x, self.y, self.t, self.v1, self.v5, sample_batch_size)
                tf_dict = {self.x_tf: x_res_batch, self.y_tf: y_res_batch, self.t_tf: t_res_batch,
                           self.v1_tf: v1_res_batch, self.v5_tf: v5_res_batch}
#                self.sess.run(self.train_op_Adam_v1, tf_dict) #uncomment if Adam sought
                self.optimizer_v1.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1])
#                self.sess.run(self.train_op_Adam_v5, tf_dict) #uncomment if Adam sought
                self.optimizer_v5.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss5])
#                self.sess.run(self.train_op_Adam_f, tf_dict) #uncomment if Adam sought
                self.optimizer_f.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1, self.loss5, self.lossf1, self.lossf5],
                                        loss_callback = self.callback)
                if it % 10 == 0:
                    loss_v1_value, loss_v5_value, loss_f1_value, loss_f5_value = self.sess.run([self.loss1, self.loss5, self.lossf1, self.lossf5], tf_dict)
                    self.loss_v1_log.append(loss_v1_value)
                    self.loss_v5_log.append(loss_v5_value)
                    self.loss_f1_log.append(loss_f1_value)
                    self.loss_f5_log.append(loss_f5_value)
                self.current_time = time.time() - self.start_time
        except KeyboardInterrupt:
            print('Externally stopped via keyboard')
            raise
    def predict(self, x_star, y_star, t_star): 
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        v1_star = self.sess.run(self.v1_pred, tf_dict)
        v5_star = self.sess.run(self.v5_pred, tf_dict)
        PINN_v2_star = self.sess.run(self.PINN_v2_pred, tf_dict)
        PINN_v3_star = self.sess.run(self.PINN_v3_pred, tf_dict)
        PINN_v4_star = self.sess.run(self.PINN_v4_pred, tf_dict) 
        return v1_star, v5_star, PINN_v2_star, PINN_v3_star, PINN_v4_star

model = PhysicsInformedNN(x_train, y_train, t_train, v1_train, v5_train, layers)
timelen_end = timelen_end*60.*60.
Nfeval = 1 
model.train(timelen_end)
 
###### ---- code for plotting ---- ######
import math 
len_loop_t = len(np.unique(x_t))
len_loop_x = len(np.unique(x_x))
len_loop_y = len(np.unique(x_y))
len_loop_z = len(np.unique(x_z)) 
    
import matplotlib.pyplot as plt

N_time = int(1)
len_skip = len_loop_x*len_loop_y*len_loop_z
len_2d = len_loop_x*len_loop_y
X0 = x_x[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
X1 = x_y[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
X2 = x_z[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
X3 = x_t[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
colormap = 'inferno'
output_model = model.predict(X0,X1,X3)

save_figs_path = save_directory

a0 = a = 0.22                       #.minor radius (m)
R0 = 0.68 + a0                      #.major radius (m)
Te0    = 25.0                       #.Electron temperature (eV).
Ti0    = 25.0                       #.Ion temperature (eV).
n0     = 5e19                       #.Plasma density (m^-3).
B0     = (5.0*0.68)/(0.68 + 0.22)   #.Magnetic field (T) on centre of simulation domain in SOL
mime   = 3672.3036                  #.Mass ratio = m_i/m_e.
Z      = 1                          #.Ionization level.

u     = 1.660539040e-27             #.atomic mass unit (kg).
m_H   = 1.007276*u                  #.Hydrogen ion (proton) mass (kg).
mu    = 2.0 #.39.948                #.m_i/m_proton.
m_D   = mu*m_H                      #.ion (singly ionized?) mass (kg).
m_i   = m_D                       #.ion mass (kg).
m_e   = 0.910938356e-30             #.electron mass (kg).
c     = 299792458.0                 #.speed of light (m/s).
e     = 1.60217662e-19              #.electron charge (C).

cse0   = np.sqrt(e*Te0/m_i)         #.sound speed (m/s). - Te
csi0   = np.sqrt(e*Ti0/m_i)         #.sound speed (m/s). - Ti

tRef   = np.sqrt((R0*a)/2.0)/cse0 
factor_space = 100.0*a0 #to get in centimetres

var = y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
y_plot = []
y_plot.append(var) 
y_plot.append(output_model[0])

x_min_plot = factor_space*min(X0)[0]
x_max_plot = factor_space*max(X0)[0]
y_min_plot = factor_space*min(X1)[0]
y_max_plot = factor_space*max(X1)[0] 

y_plot = []
var = y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
y_plot.append(var)
var = y_Te[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
y_plot.append(var)

refValmult = [n0,Te0]
i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    im = ax.scatter(factor_space*X0-x_min_plot,factor_space*X1,c=refValmult[i]*y_plot[i],cmap='YlOrRd_r')
    fig.colorbar(im, ax=axes[i])
    ax.set_xlim(x_min_plot-x_min_plot,x_max_plot-x_min_plot)
    ax.set_ylim(y_min_plot,y_max_plot)
    if i%2 == 0:
        ax.set_ylabel('y (cm)')
        ax.set_title('Observed electron density: $n_e$ (m$^{-3}$)')
    if i%2 != 0:
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Observed electron temperature: $T_e$ (eV)')
    i = i + 1

fig.subplots_adjust(right=0.8)
plt.subplots_adjust(hspace=0.4)
plt.savefig(str(save_figs_path)+'redorng_both_n_e_T_e.png')
plt.savefig(str(save_figs_path)+'redorng_both_n_e_T_e.eps')
plt.show()

phi_norm = B0*(a**2)/tRef
y_phi = full_vars_array[:,9]

xlim_min = factor_space*min(X0)[0]
xlim_max = factor_space*max(X0)[0]
ylim_min = factor_space*min(X1)[0]
ylim_max = factor_space*max(X1)[0]

inds = np.where((factor_space*X0[:,0] > xlim_min) & (factor_space*X0[:,0] < xlim_max))[0]

y_plot = []
var = y_phi[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
y_plot.append(var)
y_plot.append(output_model[2])

i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    im = ax.scatter(factor_space*X0[inds]-x_min_plot,factor_space*X1[inds],c=phi_norm*y_plot[i][inds],cmap=colormap)
    ax.set_xlim(xlim_min-x_min_plot,xlim_max-x_min_plot)
    ax.set_ylim(ylim_min,ylim_max)
    fig.colorbar(im, ax=axes[i])
    if i%2 == 0:
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Target electric potential: $\phi$ (V)')
    if i%2 != 0:
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Predicted electric potential: $\phi$ (V)')
    i = i + 1

fig.subplots_adjust(right=0.8) 
plt.subplots_adjust(hspace=0.4)
plt.savefig(str(save_figs_path)+'phi.png')
plt.savefig(str(save_figs_path)+'phi.eps')
plt.show()



tot_e_field_true = []
i = 0
while i < len_loop_y:
    ind_start = i*len_loop_x
    e_field_true = np.gradient(y_plot[0][ind_start:ind_start+len_loop_x][:,0],X0[ind_start:ind_start+len_loop_x][:,0])
    tot_e_field_true.append(e_field_true)
    i = i + 1

tot_e_field_true = np.hstack(tot_e_field_true)

tot_e_field_pred = []
i = 0
while i < len_loop_y:
    ind_start = i*len_loop_x
    e_field_pred = np.gradient(y_plot[1][ind_start:ind_start+len_loop_x][:,0],X0[ind_start:ind_start+len_loop_x][:,0])
    tot_e_field_pred.append(e_field_pred)
    i = i + 1

tot_e_field_pred = np.hstack(tot_e_field_pred)

y_plot = [] 
y_plot.append(tot_e_field_true)
y_plot.append(tot_e_field_pred)
 
i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    im = ax.scatter(factor_space*X0[inds]-x_min_plot,factor_space*X1[inds],c=-phi_norm*y_plot[i][inds]/a0,cmap=colormap)#norm=norm
    ax.set_xlim(xlim_min-x_min_plot,xlim_max-x_min_plot)
    ax.set_ylim(ylim_min,ylim_max)
    fig.colorbar(im, ax=axes[i])
    if i%2 == 0:
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Target electric field: $E_r$ (V/m)')
    if i%2 != 0:
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Predicted electric field: $E_r$ (V/m)')
    i = i + 1

fig.subplots_adjust(right=0.8) 
plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(vspace=0.2)
plt.savefig(str(save_figs_path)+'E_r_lim.png')
plt.savefig(str(save_figs_path)+'E_r_lim.eps')
plt.show()


E1 = -phi_norm*y_plot[0][inds]/a0
E2 = -phi_norm*y_plot[1][inds]/a0
print('Average electric field absolute error is: ')
print(np.mean(np.abs(E1 - E2)))

phi_norm = B0*(a**2)/tRef
y_phi = full_vars_array[:,9]

xlim_min = factor_space*min(X0)[0] 
xlim_max = factor_space*max(X0)[0] 
ylim_min = factor_space*min(X1)[0] 
ylim_max = factor_space*max(X1)[0] 

inds = np.where((factor_space*X0[:,0] > xlim_min) & (factor_space*X0[:,0] < xlim_max))[0]

y_plot = []
var = y_phi[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
y_plot.append(var)
y_plot.append(output_model[2])

fig, ax1 = plt.subplots(figsize=(12.,6.25))
x_line = factor_space*X0[inds]-x_min_plot
len_x_line = int(len(x_line)/len_loop_y)
y_line_full = factor_space*X1[inds]
y_line = int(len_loop_y/2.) #must be less than len_loop_y; selecting point approximately halfway
x_plot_1d = x_line[y_line*len_x_line:y_line*len_x_line+len_x_line]
y_plot_1d = y_line_full[y_line*len_x_line:y_line*len_x_line+len_x_line]
phi_plot_1d_actual = phi_norm*y_plot[0][inds][y_line*len_x_line:y_line*len_x_line+len_x_line]
phi_plot_1d_pred = phi_norm*y_plot[1][inds][y_line*len_x_line:y_line*len_x_line+len_x_line]
ax1.plot(x_plot_1d,phi_plot_1d_actual, color = 'r', label = 'Target')
ax1.set_ylabel(r'Target $\phi$ (V)', color = 'r', fontsize=20, labelpad=10)
ax1.tick_params(axis='y', labelcolor='r', labelsize=20)
ax1.set_xlabel('x (cm)', fontsize=20, labelpad=5) 
ax1.tick_params(axis='x', labelcolor='k', labelsize=20) 
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel(r'Predicted $\phi$ (V)', color='k', fontsize=20, labelpad=-5)
ax2.tick_params(axis='y', labelcolor='k', labelsize=20)
ax2.plot(x_plot_1d,phi_plot_1d_pred, color = 'k', label = 'Prediction')
ax1.set_ylim(80.,490.)
ax2.set_ylim(-125.,285.) 
plt.subplots_adjust(top=1.0)
plt.savefig(str(save_figs_path)+'1d_phi_char.png')
plt.savefig(str(save_figs_path)+'1d_phi_char.eps')
fig.tight_layout()
plt.show()

#print(100.0*R0*X2[inds][y_line*len_x_line:y_line*len_x_line+len_x_line])
#This is z-position in cm
#tRef*(10.**6.)*X3[inds][y_line*len_x_line:y_line*len_x_line+len_x_line]
#This time in microseconds

#Boltzmann and Neoclassical
#setting phi = 0 at n_max
den_array = n0*y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)] #arrays of den and Te are in physical units
n0_ref = np.max(den_array) #this is n0_ref NOT n0 which is reference density
Te_array = var = Te0*y_Te[int(N_time*len_skip):int(N_time*len_skip + len_2d)] #in units of eV

phi_array = np.log(den_array/n0_ref)*Te_array*e/e #e cancels since conversion from eV to J in numerator
phi_norm = B0*(a**2)/tRef
phi_plot = phi_array/phi_norm

y_Ti = h5f['y_Ti'].value
Ti_array = var = Ti0*y_Ti[int(N_time*len_skip):int(N_time*len_skip + len_2d)] #in units of eV

grad_n_field_true = []
i = 0
while i < len_loop_y:
    ind_start = i*len_loop_x
    grad_n_field_part = np.gradient(den_array[ind_start:ind_start+len_loop_x][:,0],a*X0[ind_start:ind_start+len_loop_x][:,0])
    grad_n_field_true.append(grad_n_field_part)
    i = i + 1

grad_n_field_true = np.hstack(grad_n_field_true)

grad_Ti_field_true = []
i = 0
while i < len_loop_y:
    ind_start = i*len_loop_x
    grad_Ti_field_part = np.gradient(Ti_array[ind_start:ind_start+len_loop_x][:,0],a*X0[ind_start:ind_start+len_loop_x][:,0])
    grad_Ti_field_true.append(grad_Ti_field_part)
    i = i + 1

grad_Ti_field_true = np.hstack(grad_Ti_field_true)

den_array_true = np.hstack(den_array)
Ti_array_true = np.hstack(Ti_array)
Er_ion_force_balance = (Ti_array_true*grad_n_field_true + den_array_true*grad_Ti_field_true)/(den_array_true)

den_array = n0*y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
n0_ref = np.max(den_array) #this is n0_ref NOT n0 which is reference density
Te_array = var = Te0*y_Te[int(N_time*len_skip):int(N_time*len_skip + len_2d)] #in units of eV

phi_array = np.log(den_array/n0_ref)*Te_array*e/e #e cancels since conversion from eV to J in numerator
phi_norm = B0*(a**2)/tRef
phi_plot = phi_array/phi_norm

ion_force_balance = 1. #ion force balance is with +Z since q in denominator is positive, not negative
y_plot = []
y_plot.append(phi_plot)
y_plot.append(ion_force_balance*Er_ion_force_balance/phi_norm)


refs = []
refs.append(phi_norm)
refs.append(phi_norm/a0)

i = 0
fig, axes = plt.subplots(nrows=2, ncols=1)
for ax in axes.flat:
    im = ax.scatter(factor_space*X0[inds]-x_min_plot,factor_space*X1[inds],c=refs[i]*y_plot[i][inds],cmap=colormap)
    ax.set_xlim(xlim_min-x_min_plot,xlim_max-x_min_plot)
    ax.set_ylim(ylim_min,ylim_max)
    fig.colorbar(im, ax=axes[i])
    if i == 0:
        ax.set_ylabel('y (cm)')
        ax.set_title(r'Boltzmann potential: $\phi$ (V)')
    if i == 1:
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title('Neoclassical electric field: $E_r$ (V/m)')
    i = i + 1

fig.subplots_adjust(right=0.8) 
plt.subplots_adjust(hspace=0.4) 
plt.savefig(str(save_figs_path)+'Boltz_neo.png')
plt.savefig(str(save_figs_path)+'Boltz_neo.eps')
plt.show()

#normalized losses multiplied by init_weight
loss_v1 = model.loss_v1_log
loss_v5 = model.loss_v5_log
loss_f1 = model.loss_f1_log
loss_f5 = model.loss_f5_log 

fig_2 = plt.figure(2)
ax = fig_2.add_subplot(1, 1, 1)
ax.plot(loss_f5, label='$\mathcal{L(f)}_{T_e}$') #/init_weight_Te
ax.plot(loss_f1, label='$\mathcal{L(f)}_{n_e}$') #/init_weight_den
ax.plot(loss_v5, label='$\mathcal{L}_{T_e}$')    #/init_weight_Te
ax.plot(loss_v1, label='$\mathcal{L}_{n_e}$')    #/init_weight_den
ax.set_yscale('log')
ax.set_xlabel('iterations')
ax.set_ylabel('Loss') 
plt.legend()
plt.tight_layout()
plt.show()