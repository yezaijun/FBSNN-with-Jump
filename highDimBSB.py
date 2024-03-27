import numpy as np
import tensorflow as tf
import logging
import scipy.stats as ss

import Tools 
from Network import *
from SolverNN import FBSNN

class highDimBSB(FBSNN):
    def __init__(self, T, sample_size, num_time_interval, dim,model):
        super().__init__(T, sample_size, num_time_interval, dim, model)
        self.lam = 0.3
        self.dim_tf = tf.cast(self.dim,dtype=tf.float64)
        self.sigma = 0.4
        self.r = 0.05
        self.constant = self.r + self.sigma**2
        self.T_tf = tf.cast(self.terminal_time,dtype=tf.float64)
        self.mu_normal = 0.02
        self.sigma_normal  = 0.01
        self.mu_sig_2 = self.mu_normal**2 + self.sigma_normal**2
        self.delta_t = 0.02

        self.jump_time_max = int(ss.poisson.ppf(0.99999999, self.lam*self.delta_t))
 
    def Forward_SDE(self,t,x,dt,dw,jump):
        x1 = x + self.sigma * x * dw  + jump + self.r*x*dt
        return x1
    
    def Backward_SDE(self,t,x,dt,dw,y,dy_dx,jump):
        y1 = y + self.phi_tf(t,x,y,dy_dx) * dt + self.sigma* tf.reduce_sum(dy_dx * x * dw,axis=1,keepdims=True) + jump
        return y1
    
    def phi_tf(self, t, X, Y, Z):
        return self.r * Y + self.lam *  tf.exp(self.constant*(self.T_tf - t)) * self.mu_sig_2

    
    def Terminal_condition(self,t,x):
        return tf.reduce_mean(x**2,axis=1,keepdims=True)

    def Solution(self,t,x,training=False):
        x2_mean = tf.reduce_mean(x**2,axis=1,keepdims=True)
        x_mean = tf.reduce_mean(x,axis=1,keepdims=True)
        return {
            'y' : tf.exp(self.constant*(self.T_tf - t)) * x2_mean,
            'u' : self.lam * tf.exp(self.constant*(self.T_tf - t)) *( 2*self.mu_normal*x_mean + self.mu_sig_2),
            'dy_dx':2 * tf.exp(self.constant*(self.T_tf - t)) * x/self.dim_tf,
        }
    
    def model_approximate(self,t,x,training=False):
        network_input = tf.concat([t,x], axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(network_input)
            solution = self.model(network_input, training)
            y = solution
        grad = tape.gradient(y, network_input)
        grad_x = grad[:,1:]
        del tape
        return {
            'y':y,
            'u':self.lam*self.mu_normal*tf.reduce_sum(grad_x,axis=1,keepdims=True), 
            'dy_dx':grad_x
            } 

    def Element_jump(self,num_sample):
        P_sample = ss.poisson.rvs(self.lam*self.delta_t,size=num_sample)
        output = np.zeros((num_sample, self.jump_time_max, self.dim))
        for idx,k in enumerate(P_sample):
            k_create = min(k,self.jump_time_max)
            if k_create != 0:
                output[idx,:k_create,:self.dim] = np.random.normal(self.mu_normal, self.sigma_normal,(k_create,self.dim))
        return output

    def Forward_jump(self,t,x,dt,dw,element_jump):
        Jumps_sample = np.sum( element_jump , axis=1 ) - dt * self.lam * self.mu_normal
        return Jumps_sample
    
    def jump_bate_fun(self,t,x,jump):
        return x + jump

if __name__ == "__main__":
    # init logging
    logging.basicConfig(
        format = '%(levelname)-6s %(message)s',
        level=logging.INFO
    )

    PDE_name = "HighDimBSB"
    T = 1
    sample_size = 100
    num_time_interval =50
    dim = 100
    x_initial = np.ones((sample_size,dim))

    model = FeedForwardNet([128,128,128,128,128],tf.nn.leaky_relu,output_dim=1)
    mySolver = highDimBSB(T, sample_size, num_time_interval, dim, model)
    logging.info(f'Begin to solve {PDE_name}')

    # 结果保存
    path_dir = Tools.init_saving_path(rootDir = "./"+PDE_name,
                                      pde_config = mySolver.get_config(),
                                      model_config= mySolver.model.get_config())
    logging.info(f'logfile Dir:{path_dir["rootDir"]}')

    # 检查损失函数是否合理
    # Tools.visual_check(mySolver,sample_size=2)
    mySolver.check_loss_on_solution(sample_size = 1000)

    # 定义学习率调度器
    boundaries = [2000,4000]
    values = [1e-3, 1e-4,1e-5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_fn)

    checkpoint = Tools.CheckPoint(model=mySolver.model,dir=path_dir['checkpoint_path'])
    
    # 训练模型
    mySolver.train(num_iterations=5000,optimizer=optimizer)

    # 保存结果
    Tools.dump_model(mySolver.model,path_dir["model_path"])
    Tools.save_trainHistory(mySolver.training_history,path_dir["training_history_file"])
