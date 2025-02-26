"""Physics-Informed Neural Network implementation."""

import tensorflow.compat.v1 as tf  # Use TF 1.x compatibility mode
tf.disable_v2_behavior()  # Disable TF 2.x behavior
import numpy as np
from typing import Dict, List, Tuple, Callable
from utils.constants import *
from config.settings import *
from scipy.optimize import minimize
import os

class PINN:
    def __init__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, 
                 v1: np.ndarray, v5: np.ndarray, layers: List[int], 
                 use_pde: bool = True):
        """Initialize PINN model."""
        # Data initialization
        self.x, self.y, self.t = x, y, t
        self.v1, self.v5 = v1, v5
        self.layers = layers or LAYERS
        self.use_pde = use_pde
        
        # Get normalization constants
        self.diff_norms = DIFF_NORMS
        
        # Setup networks
        X = np.concatenate([x, y, t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        
        # Initialize network weights
        self.weights_v1, self.biases_v1 = self.initialize_NN(self.layers)
        self.weights_v5, self.biases_v5 = self.initialize_NN(self.layers)
        
        if use_pde:
            self.weights_v2, self.biases_v2 = self.initialize_NN(self.layers)
            self.weights_v3, self.biases_v3 = self.initialize_NN(self.layers)
            self.weights_v4, self.biases_v4 = self.initialize_NN(self.layers)
            self.loss_history = {'v1': [], 'v5': [], 'f1': [], 'f5': []}
        else:
            self.loss_history = {'v1': [], 'v5': []}

        # Setup TF session and training
        self.setup_session()
        self.setup_placeholders()
        self.setup_optimizers()
        
        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        # Add saver
        self.saver = tf.train.Saver()

    def setup_session(self):
        """Setup TensorFlow session configuration."""
        # Allow memory growth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Create session with config
        self.sess = tf.Session(config=config)
        
        # Initialize all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def neural_net(self, X: tf.Tensor, weights: List[tf.Variable], 
                  biases: List[tf.Variable]) -> tf.Tensor:
        """Forward pass through network with proper normalization."""
        # Input normalization
        X_d0 = max(self.ub[0] - self.lb[0], 1e-6)
        X_d1 = max(self.ub[1] - self.lb[1], 1e-6)
        X_d2 = max(self.ub[2] - self.lb[2], 1e-6)
        X_d = np.array([X_d0, X_d1, X_d2])
        H = 2.0*(X - self.lb)/X_d - 1.0
        
        # Forward pass
        num_layers = len(weights) + 1
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            
        W = weights[-1]
        b = biases[-1]
        return tf.add(tf.matmul(H, W), b)

    def net_plasma(self, x: tf.Tensor, y: tf.Tensor, t: tf.Tensor):
        """Compute plasma variables and PDE terms."""
        # Core network outputs
        v1 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v1, self.biases_v1)
        v5 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v5, self.biases_v5)
        
        if not self.use_pde:
            return v1, v5, None, None, None, None, None
            
        # Additional networks for PDE terms
        v2 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v2, self.biases_v2)
        v3 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v3, self.biases_v3)
        v4 = self.neural_net(tf.concat([x,y,t], 1), self.weights_v4, self.biases_v4)

        # Compute derivatives
        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v2_x = tf.gradients(v2, x)[0]
        v2_y = tf.gradients(v2, y)[0]
        v5_t = tf.gradients(v5, t)[0]
        v5_x = tf.gradients(v5, x)[0]
        v5_y = tf.gradients(v5, y)[0]

        # Physics calculations
        B = (0.22+0.68)/(0.68 + 0.22 + MINOR_RADIUS*x)
        pe = v1*v5
        pe_y = tf.gradients(pe, y)[0]
        jp = v1*((TAU_T**0.5)*v4 - v3)

        # log form of variables for diffusion
        lnn = tf.log(v1)
        lnTe = tf.log(v5)
        
        # Compute up to 4th derivatives
        D_lnn, D_lnTe = self.compute_high_order_derivs(lnn, lnTe, x, y)
        
        # Source terms
        S_n, S_Ee = self.compute_source_terms(x, v1, v5)
        
        # Compute residuals
        f_v1, f_v5 = self.compute_residuals(v1, v5, v1_t, v1_x, v1_y, v2_x, v2_y,
                                            v5_t, v5_x, v5_y, B, pe_y, jp, pe,
                                            D_lnn, D_lnTe, S_n, S_Ee)
        
        return v1, v5, v2, v3, v4, f_v1, f_v5

    def compute_high_order_derivs(self, lnn, lnTe, x, y):
        """Compute derivatives up to 4th order for diffusion."""
        lnn_x = tf.gradients(lnn, x)[0]
        lnn_xx = tf.gradients(lnn_x, x)[0]
        lnn_xxx = tf.gradients(lnn_xx, x)[0]
        lnn_xxxx = tf.gradients(lnn_xxx, x)[0]
        lnn_y = tf.gradients(lnn, y)[0]
        lnn_yy = tf.gradients(lnn_y, y)[0]
        lnn_yyy = tf.gradients(lnn_yy, y)[0]
        lnn_yyyy = tf.gradients(lnn_yyy, y)[0]

        # Calculate same for temperature
        lnTe_x = tf.gradients(lnTe, x)[0]
        lnTe_xx = tf.gradients(lnTe_x, x)[0]
        lnTe_xxx = tf.gradients(lnTe_xx, x)[0]
        lnTe_xxxx = tf.gradients(lnTe_xxx, x)[0]
        lnTe_y = tf.gradients(lnTe, y)[0]
        lnTe_yy = tf.gradients(lnTe_y, y)[0]
        lnTe_yyy = tf.gradients(lnTe_yy, y)[0]
        lnTe_yyyy = tf.gradients(lnTe_yyy, y)[0]

        # Original diffusion terms exactly as in old code
        Dx_lnn = -((50./self.diff_norms['DiffX_norm'])**2.)*lnn_xxxx
        Dy_lnn = -((50./self.diff_norms['DiffY_norm'])**2.)*lnn_yyyy
        D_lnn = (Dx_lnn + Dy_lnn)
        
        Dx_lnTe = -((50./self.diff_norms['DiffX_norm'])**2.)*lnTe_xxxx
        Dy_lnTe = -((50./self.diff_norms['DiffY_norm'])**2.)*lnTe_yyyy
        D_lnTe = (Dx_lnTe + Dy_lnTe)

        return D_lnn, D_lnTe

    def compute_source_terms(self, x, v1, v5):
        """Compute source terms with conditions."""
        # Get dynamic batch size from input tensor
        batch_size = tf.shape(x)[0]
        
        S_n = N_SRC_A*tf.exp(-(x - X_SRC)*(x - X_SRC)/(2.*SIG_SRC**2))
        S_Ee = ENER_SRC_A*tf.exp(-(x - X_SRC)*(x - X_SRC)/(2.*SIG_SRC**2))

        # Apply source term conditions using dynamic batch size
        cond1Sn = tf.greater(S_n[:,0], 0.01*tf.ones(batch_size))
        S_n = tf.where(cond1Sn, S_n[:,0], 0.001*tf.ones(batch_size))
        cond1SEe = tf.greater(S_Ee[:,0], 0.01*tf.ones(batch_size))
        S_Ee = tf.where(cond1SEe, S_Ee[:,0], 0.001*tf.ones(batch_size))
        
        cond2Sn = tf.greater(x, X_SRC*tf.ones(batch_size))
        S_n = tf.where(cond2Sn[:,0], S_n, 0.5*tf.ones(batch_size))
        cond2SEe = tf.greater(x, X_SRC*tf.ones(batch_size))
        S_Ee = tf.where(cond2SEe[:,0], S_Ee, 0.5*tf.ones(batch_size))
        
        cond4Sn = tf.greater(v1[:,0], 5.0*tf.ones(batch_size))
        S_n = tf.where(cond4Sn, 0.0*tf.ones(batch_size), S_n)
        cond4SEe = tf.greater(v5[:,0], 1.0*tf.ones(batch_size))
        S_Ee = tf.where(cond4SEe, 0.0*tf.ones(batch_size), S_Ee)

        return S_n, S_Ee

    def compute_residuals(self, v1, v5, v1_t, v1_x, v1_y, v2_x, v2_y,
                          v5_t, v5_x, v5_y, B, pe_y, jp, pe, D_lnn, D_lnTe, S_n, S_Ee):
        """Compute PDE residuals."""
        f_v1 = v1_t + (1./B)*(v2_y*v1_x - v2_x*v1_y) - (-EPS_R*(v1*v2_y - ALPHA_D*pe_y) + S_n + v1*D_lnn)
        f_v5 = v5_t + (1./B)*(v2_y*v5_x - v2_x*v5_y) - v5*(5.*EPS_R*ALPHA_D*v5_y/3. +
                (2./3.)*(-EPS_R*(v2_y - ALPHA_D*pe_y/v1) +
                (1./v1)*(0.71*EPS_V*(0.0) + ETA*jp*jp/(v5*MASS_RATIO))) + (2./(3.*pe))*(S_Ee) + D_lnTe)

        return f_v1, f_v5

    def train_step(self):
        """Perform one training step."""
        # Get batch
        idx_batch = np.random.choice(len(self.x), SAMPLE_BATCH_SIZE, replace=False)
        tf_dict = {
            self.x_tf: self.x[idx_batch],
            self.y_tf: self.y[idx_batch],
            self.t_tf: self.t[idx_batch],
            self.v1_tf: self.v1[idx_batch],
            self.v5_tf: self.v5[idx_batch]
        }
        
        # Run training ops
        self.sess.run([self.train_op_v1, self.train_op_v5], feed_dict=tf_dict)
        
        if self.use_pde:
            self.sess.run(self.train_op_f, feed_dict=tf_dict)
            loss_v1, loss_v5, loss_f1, loss_f5 = self.sess.run(
                [self.loss_v1, self.loss_v5, self.loss_f1, self.loss_f5],
                feed_dict=tf_dict
            )
            self.loss_history['f1'].append(loss_f1)
            self.loss_history['f5'].append(loss_f5)
            self.loss_history['v1'].append(loss_v1)
            self.loss_history['v5'].append(loss_v5)
            return {'v1': loss_v1, 'v5': loss_v5, 'f1': loss_f1, 'f5': loss_f5,
                   'total': loss_v1 + loss_v5 + loss_f1 + loss_f5}
        else:
            loss_v1, loss_v5 = self.sess.run(
                [self.loss_v1, self.loss_v5],
                feed_dict=tf_dict
            )
            self.loss_history['v1'].append(loss_v1)
            self.loss_history['v5'].append(loss_v5)
            return {'v1': loss_v1, 'v5': loss_v5, 'total': loss_v1 + loss_v5}

    def setup_optimizers(self):
        """Setup Adam optimizers for training.
        
        Note: The original paper used L-BFGS optimization (tf.contrib.opt.ScipyOptimizerInterface).
        However, due to TF2.x compatibility issues, we now use Adam. To exactly reproduce 
        paper results, you would need to use TF1.x with L-BFGS optimization.
        """
        # Data fitting optimizers
        self.optimizer_v1 = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_v1 = self.optimizer_v1.minimize(
            self.loss_v1,
            var_list=self.weights_v1 + self.biases_v1
        )
        
        self.optimizer_v5 = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_v5 = self.optimizer_v5.minimize(
            self.loss_v5,
            var_list=self.weights_v5 + self.biases_v5
        )

        if self.use_pde:
            self.optimizer_f = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op_f = self.optimizer_f.minimize(
                self.loss_f1 + self.loss_f5,
                var_list=(self.weights_v2 + self.biases_v2 + 
                         self.weights_v3 + self.biases_v3 +
                         self.weights_v4 + self.biases_v4)
            )
    
    # Utility methods
    def initialize_NN(self, layers: List[int]) -> Tuple[List[tf.Variable], List[tf.Variable]]:
        """Initialize neural network weights and biases."""
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def setup_placeholders(self):
        """Setup TensorFlow placeholders for inputs and outputs."""
        # Input placeholders
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Output placeholders
        self.v1_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v5_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Network predictions and residuals
        self.v1_pred, self.v5_pred, self.v2_pred, self.v3_pred, self.v4_pred,\
        self.f_v1_pred, self.f_v5_pred = self.net_plasma(self.x_tf, self.y_tf, self.t_tf)
        
        # Define losses
        self.loss_v1 = tf.reduce_mean(INIT_WEIGHT_DEN * tf.square(self.v1_tf - self.v1_pred))
        self.loss_v5 = tf.reduce_mean(INIT_WEIGHT_TE * tf.square(self.v5_tf - self.v5_pred))
        
        if self.use_pde:
            self.loss_f1 = tf.reduce_mean(INIT_WEIGHT_DEN * tf.square(self.f_v1_pred))
            self.loss_f5 = tf.reduce_mean(INIT_WEIGHT_TE * tf.square(self.f_v5_pred))
            self.loss = self.loss_v1 + self.loss_v5 + self.loss_f1 + self.loss_f5
        else:
            self.loss = self.loss_v1 + self.loss_v5

    def predict(self, x_star: np.ndarray, y_star: np.ndarray, 
                t_star: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using trained model."""
        tf_dict = {
            self.x_tf: x_star,
            self.y_tf: y_star,
            self.t_tf: t_star
        }
        
        if self.use_pde:
            v1, v5, v2, v3, v4, _, _ = self.sess.run(
                [self.v1_pred, self.v5_pred, self.v2_pred, 
                 self.v3_pred, self.v4_pred, self.f_v1_pred, self.f_v5_pred],
                feed_dict=tf_dict
            )
            return {
                'v1': v1, 'v2': v2, 'v3': v3, 
                'v4': v4, 'v5': v5
            }
        else:
            v1, v5 = self.sess.run(
                [self.v1_pred, self.v5_pred],
                feed_dict=tf_dict
            )
            return {
                'v1': v1, 'v5': v5
            }

    def xavier_init(self, size: List[int]) -> tf.Variable:
        """Xavier initialization for network weights."""
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim], 
                mean=0.0,
                stddev=xavier_stddev
            ),
            dtype=tf.float32
        )

    def save(self, save_path: str):
        """Save model using TensorFlow's Saver."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        checkpoint_path = os.path.join(save_path, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path)
        
        # Save metadata separately (since it's not part of TF graph)
        metadata = {
            'layers': self.layers,
            'use_pde': self.use_pde,
            'lb': self.lb,
            'ub': self.ub,
            'diff_norms': self.diff_norms,
            'loss_history': self.loss_history
        }
        
        metadata_path = os.path.join(save_path, 'metadata.npz')
        np.savez(metadata_path, **metadata)

    def load(self, load_path: str):
        """Load model from checkpoint."""
        # Load model weights
        checkpoint_path = os.path.join(load_path, 'model.ckpt')
        self.saver.restore(self.sess, checkpoint_path)
        
        # Load metadata
        metadata = dict(np.load(os.path.join(load_path, 'metadata.npz'), allow_pickle=True))
        
        # Process metadata fields with proper type conversion
        if isinstance(metadata['layers'], np.ndarray):
            if metadata['layers'].dtype == np.dtype('O'):
                self.layers = metadata['layers'].item()
            else:
                self.layers = metadata['layers'].tolist()
        else:
            self.layers = metadata['layers']
            
        self.use_pde = bool(metadata['use_pde'].item() if isinstance(metadata['use_pde'], np.ndarray) else metadata['use_pde'])
        self.lb = metadata['lb']
        self.ub = metadata['ub']
        self.diff_norms = metadata['diff_norms'].item() if isinstance(metadata['diff_norms'], np.ndarray) else metadata['diff_norms']
        self.loss_history = metadata['loss_history'].item() if isinstance(metadata['loss_history'], np.ndarray) else metadata['loss_history']
