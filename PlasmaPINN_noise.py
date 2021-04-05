# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:26:54 2021

@author: mathewsa
"""

import h5py
import numpy as np

save_directory = '/home/mathewsa/Plasma-PINN_manuscript_v0/noise/' #path to find files and save them

N_outputs = 1 #number of predictions made by each PINN
timelen_end = 20.0 #training time (hours)

mean = 1.0 #since multiplying true data, 1.0 implies no systematic offset in addition to Gaussian noise
std_err = 0.25

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

data_file = str(save_directory)+'PlasmaPINN_data_inputs_paper_noise.h5' 
h5f = h5py.File(data_file, "r")
x_x = h5f['x_x'].value 
x_y = h5f['x_y'].value  
x_z = h5f['x_z'].value  
x_t = h5f['x_t'].value   
y_den = h5f['y_den'].value 

init_weight_den = (1./np.median(np.abs(y_den))) 
frac_train = 1.0
N_train = int(frac_train*len(y_den))
idx = np.random.choice(len(y_den), N_train, replace=False)

x_train = x_x[idx,:]
y_train = x_y[idx,:]
z_train = x_z[idx,:]
t_train = x_t[idx,:]
v1_train = y_den[idx,:]

sample_batch_size = int(2500)

import time
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class PhysicsInformedNN_with_f1_opt:
    def __init__(self, x, y, t, v1, layers):
        X = np.concatenate([x, y, t], 1) 
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.v1 = v1
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
        self.v1_pred,\
        self.PINN_v2_pred, self.PINN_v3_pred, self.PINN_v4_pred, self.PINN_v5_pred,\
        self.f_v1_pred, self.f_v5_pred = self.net_plasma(self.x_tf, self.y_tf, self.t_tf)
        self.loss1 = tf.reduce_mean(1.*init_weight_den*tf.square(self.v1_tf - self.v1_pred))
        self.lossf1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.f_v1_pred)) 
        self.optimizer_v1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(self.loss1 + self.lossf1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1+self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4+self.weights_v5+self.biases_v5,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.loss_v1_log = []
        self.loss_f1_log = []
        self.train_op_Adam_v1 = self.optimizer_Adam.minimize(self.loss1, var_list=self.weights_v1+self.biases_v1)
        self.train_op_Adam_f = self.optimizer_Adam.minimize(self.loss1 + self.lossf1, var_list=self.weights_v1+self.biases_v1+self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4+self.weights_v5+self.biases_v5)
        init = tf.global_variables_initializer() # Initialize Tensorflow variables
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
    def neural_net_bound(self, X, weights, biases, lower, upper):
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
        scale = upper - lower
        bound_Y = scale * tf.nn.sigmoid(Y) + lower 
        return bound_Y
    def net_plasma(self, x, y, t): 
        mi_me = 3672.3036
        eta = 63.6094
        nSrcA = 20.0
        enerSrceA = 0.001
        enerSrciA = 0.001
        xSrc = -0.15
        sigSrc = 0.01
        B = (0.22+0.68)/(0.68 + 0.22 + a0*x) #check
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
        v3 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v3, self.biases_v3, -2.0, 2.0)
        v4 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v4, self.biases_v4, -2.0, 2.0)
        v5 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v5, self.biases_v5, 0.0, 5.0)
        PINN_v2 = v2
        PINN_v3 = v3
        PINN_v4 = v4
        PINN_v5 = v5 
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
        return v1, PINN_v2, PINN_v3, PINN_v4, PINN_v5, f_v1, f_v5
    def callback(self, loss1, lossf1):
        global Nfeval
        print(str(Nfeval)+' - PDE loss in loop: %.3e, %.3e' % (loss1, lossf1))
        Nfeval += 1
    def fetch_minibatch(self, x_in, y_in, t_in, den_in, N_train_sample):
        idx_batch = np.random.choice(len(x_in), N_train_sample, replace=False)
        x_batch = x_in[idx_batch,:]
        y_batch = y_in[idx_batch,:]
        t_batch = t_in[idx_batch,:]
        v1_batch = den_in[idx_batch,:]
        return x_batch, y_batch, t_batch, v1_batch
    def train(self, timelen_end):
        self.start_time = time.time()
        self.current_time = time.time() - self.start_time
        try:
            it = 0
            while self.current_time < timelen_end:
                it = it + 1
                print('Full iteration: '+str(it))
                x_res_batch, y_res_batch, t_res_batch, v1_res_batch = self.fetch_minibatch(self.x, self.y, self.t, self.v1, sample_batch_size) # Fetch residual mini-batch
                tf_dict = {self.x_tf: x_res_batch, self.y_tf: y_res_batch, self.t_tf: t_res_batch,
                           self.v1_tf: v1_res_batch}
                self.optimizer_v1.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1])
                self.optimizer_f.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1, self.lossf1],
                                        loss_callback = self.callback)
                if it % 10 == 0:
                    loss_v1_value, loss_f1_value = self.sess.run([self.loss1, self.lossf1], tf_dict)
                    self.loss_v1_log.append(loss_v1_value)
                    self.loss_f1_log.append(loss_f1_value) 
                self.current_time = time.time() - self.start_time
        except KeyboardInterrupt:
            print('Externally stopped via keyboard')
            raise
    def predict(self, x_star, y_star, t_star): 
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        v1_star = self.sess.run(self.v1_pred, tf_dict)
        PINN_v2_star = self.sess.run(self.PINN_v2_pred, tf_dict)
        PINN_v3_star = self.sess.run(self.PINN_v3_pred, tf_dict)
        PINN_v4_star = self.sess.run(self.PINN_v4_pred, tf_dict)
        PINN_v5_star = self.sess.run(self.PINN_v5_pred, tf_dict) 
        return v1_star, PINN_v2_star, PINN_v3_star, PINN_v4_star, PINN_v5_star

noise_dens = np.random.normal(mean,std_err,len(v1_train))
noisy_dens_train = np.array([noise_dens*v1_train[:,0]]).T

model_with_f1_opt = PhysicsInformedNN_with_f1_opt(x_train, y_train, t_train, noisy_dens_train, layers)
 
timelen_end = timelen_end*60.*60.
Nfeval = 1
model_with_f1_opt.train(timelen_end)

class PhysicsInformedNN_without_f1_opt:
    def __init__(self, x, y, t, v1, layers):
        X = np.concatenate([x, y, t], 1) 
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]
        self.v1 = v1
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
        self.v1_pred,\
        self.PINN_v2_pred, self.PINN_v3_pred, self.PINN_v4_pred, self.PINN_v5_pred,\
        self.f_v1_pred, self.f_v5_pred = self.net_plasma(self.x_tf, self.y_tf, self.t_tf) 
        self.loss1 = tf.reduce_mean(1.*init_weight_den*tf.square(self.v1_tf - self.v1_pred))
        self.lossf1 = tf.reduce_mean(1.0*init_weight_den*tf.square(self.f_v1_pred))
        self.optimizer_v1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(self.loss1 + self.lossf1,
                                                                method = 'L-BFGS-B',
                                                                var_list=self.weights_v1+self.biases_v1+self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4+self.weights_v5+self.biases_v5,
                                                                options = {'maxiter': 50,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer() 
        self.loss_v1_log = []
        self.loss_f1_log = [] 
        self.train_op_Adam_v1 = self.optimizer_Adam.minimize(self.loss1, var_list=self.weights_v1+self.biases_v1)
        self.train_op_Adam_f = self.optimizer_Adam.minimize(self.loss1 + self.lossf1, var_list=self.weights_v1+self.biases_v1+self.weights_v2+self.biases_v2+self.weights_v3+self.biases_v3+self.weights_v4+self.biases_v4+self.weights_v5+self.biases_v5)
        self.saver = tf.train.Saver()
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
    def neural_net_bound(self, X, weights, biases, lower, upper):
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
        scale = upper - lower
        bound_Y = scale * tf.nn.sigmoid(Y) + lower 
        return bound_Y
    def net_plasma(self, x, y, t): 
        mi_me = 3672.3036
        eta = 63.6094
        nSrcA = 20.0
        enerSrceA = 0.001
        enerSrciA = 0.001
        xSrc = -0.15
        sigSrc = 0.01
        B = (0.22+0.68)/(0.68 + 0.22 + a0*x) #check
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
        v3 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v3, self.biases_v3, -2.0, 2.0)
        v4 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v4, self.biases_v4, -2.0, 2.0)
        v5 = self.neural_net_bound(tf.concat([x,y,t], 1), self.weights_v5, self.biases_v5, 0.0, 5.0)
        PINN_v2 = v2
        PINN_v3 = v3
        PINN_v4 = v4
        PINN_v5 = v5 
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
        return v1, PINN_v2, PINN_v3, PINN_v4, PINN_v5, f_v1, f_v5
    def callback(self, loss1, lossf1):
        global Nfeval
        print(str(Nfeval)+' - PDE loss in loop: %.3e, %.3e' % (loss1, lossf1))
        Nfeval += 1
    def fetch_minibatch(self, x_in, y_in, t_in, den_in, N_train_sample):
        idx_batch = np.random.choice(len(x_in), N_train_sample, replace=False)
        x_batch = x_in[idx_batch,:]
        y_batch = y_in[idx_batch,:]
        t_batch = t_in[idx_batch,:]
        v1_batch = den_in[idx_batch,:]
        return x_batch, y_batch, t_batch, v1_batch
    def train(self, timelen_end): 
        self.start_time = time.time()
        self.current_time = time.time() - self.start_time
        try:
            it = 0
            while self.current_time < timelen_end:
                it = it + 1
                print('Full iteration: '+str(it))
                x_res_batch, y_res_batch, t_res_batch, v1_res_batch = self.fetch_minibatch(self.x, self.y, self.t, self.v1, sample_batch_size) # Fetch residual mini-batch
                tf_dict = {self.x_tf: x_res_batch, self.y_tf: y_res_batch, self.t_tf: t_res_batch,
                           self.v1_tf: v1_res_batch}
                self.optimizer_v1.minimize(self.sess,
                                        feed_dict = tf_dict,
                                        fetches = [self.loss1, self.lossf1],
                                        loss_callback = self.callback)
                if it % 10 == 0:
                    loss_v1_value, loss_f1_value = self.sess.run([self.loss1, self.lossf1], tf_dict)
                    self.loss_v1_log.append(loss_v1_value)
                    self.loss_f1_log.append(loss_f1_value) 
                self.current_time = time.time() - self.start_time
        except KeyboardInterrupt:
            print('Externally stopped via keyboard')
            raise  
    def predict(self, x_star, y_star, t_star): 
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        v1_star = self.sess.run(self.v1_pred, tf_dict)
        PINN_v2_star = self.sess.run(self.PINN_v2_pred, tf_dict)
        PINN_v3_star = self.sess.run(self.PINN_v3_pred, tf_dict)
        PINN_v4_star = self.sess.run(self.PINN_v4_pred, tf_dict)
        PINN_v5_star = self.sess.run(self.PINN_v5_pred, tf_dict) 
        return v1_star, PINN_v2_star, PINN_v3_star, PINN_v4_star, PINN_v5_star

model_without_f1_opt = PhysicsInformedNN_without_f1_opt(x_train, y_train, t_train, noisy_dens_train, layers)

timelen_end = timelen_end*60.*60.
Nfeval = 1
model_without_f1_opt.train(timelen_end)

###### ---- code for plotting ---- ######
import math 
save_figs_path = save_directory

a0 = a = 0.22        #.minor radius (m)
R0 = 0.68 + a0        #.major radius (m)
Te0    = 25.0          #.Electron temperature (eV).
Ti0    = 25.0           #.Ion temperature (eV).
n0     = 5e19          #.Plasma density (m^-3).
B0     = (5.0*0.68)/(0.68 + 0.22)   #.Magnetic field (T) on centre of simulation domain in SOL
mime   = 3672.3036           #.Mass ratio = m_i/m_e.
Z      = 1             #.Ionization level.

u     = 1.660539040e-27     #.unifited atomic mass unit (kg).
m_H   = 1.007276*u          #.Hydrogen ion (proton) mass (kg).
mu    = 2.0 #.39.948              #.m_i/m_proton.
m_D  = mu*m_H              #.ion (singly ionized?) mass (kg).
m_i   = m_D                #.ion mass (kg).
m_e   = 0.910938356e-30     #.electron mass (kg).
c     = 299792458.0         #.speed of light (m/s).
e     = 1.60217662e-19      #.electron charge (C).

cse0   = np.sqrt(e*Te0/m_i)     #.Electron sound speed (m/s).
csi0   = np.sqrt(e*Ti0/m_i)     #.Ion sound speed (m/s).

tRef   = np.sqrt((R0*a)/2.0)/cse0 
factor_space = 100.0*a0 #to get in centimetres
 
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
 
var_actual = y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)] 
var = noisy_dens_train[int(N_time*len_skip):int(N_time*len_skip + len_2d)] 
output_model_with_f1_opt = model_with_f1_opt.predict(X0,X1,X3)  
output_model_without_f1_opt = model_without_f1_opt.predict(X0,X1,X3)  
 
x_min_plot = min(factor_space*X0)[0]
x_max_plot = max(factor_space*X0)[0]
y_min_plot = min(factor_space*X1)[0]
y_max_plot = max(factor_space*X1)[0]
nRef = 5.*(10e19)#n0

inds_plot_noisy = np.where(t_train == X3[0][0])

i = 0
fig21, axes = plt.subplots(nrows=4, ncols=1, figsize = (5,8))
for ax in axes.flat:
    ax.set_xlim(x_min_plot-x_min_plot,x_max_plot-x_min_plot)
    ax.set_ylim(y_min_plot,y_max_plot)
    if i == 0:
        im = ax.scatter(factor_space*X0-x_min_plot,factor_space*X1,c=nRef*var_actual,cmap='YlOrRd_r')
        fig21.colorbar(im, ax=axes[i])
        ax.set_title('Target density: $n_e$ (m$^{-3}$)') 
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_ylabel('y (cm)')
    if i == 1:
        im = ax.scatter(factor_space*x_train[inds_plot_noisy]-x_min_plot,factor_space*y_train[inds_plot_noisy],c=nRef*noisy_dens_train[inds_plot_noisy],cmap='YlOrRd_r',s=11.0)
        fig21.colorbar(im, ax=axes[i])
        ax.set_title('Noisy observed density: $n_e$ (m$^{-3}$)') 
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_ylabel('y (cm)')
    if i == 2:
        im = ax.scatter(factor_space*X0-x_min_plot,factor_space*X1,c=nRef*output_model_with_f1_opt[0],cmap='YlOrRd_r')#cmap='Greens')
        fig21.colorbar(im, ax=axes[i])
        ax.set_title('PINN: $n_e$ (m$^{-3}$)') 
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_ylabel('y (cm)')
    if i == 3:
        im = ax.scatter(factor_space*X0-x_min_plot,factor_space*X1,c=nRef*output_model_without_f1_opt[0],cmap='YlOrRd_r')#cmap='Greens')
        fig21.colorbar(im, ax=axes[i])
        ax.set_title('Classical NN: $n_e$ (m$^{-3}$)')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
    i = i + 1

fig21.subplots_adjust(right=0.5)
plt.subplots_adjust(hspace=0.01)
fig21.tight_layout(pad=1.0) 
plt.show(fig21)

#to learn phi well, further training of model without v1 weights and biases is necessary