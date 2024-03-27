import numpy as np
import tensorflow as tf
import logging
import scipy.stats as ss

import Tools
from Network import *
from SolverNN import FBSNN

class oneDimPureJump(FBSNN):
    def __init__(self, T, sample_size, num_time_interval, dim,model):
        super().__init__(T, sample_size, num_time_interval, dim,  model)
        self.lam = 0.3
        self.mu = 0.4
        self.sigma = 0.25

        self.constant = tf.cast(self.lam*(tf.exp(self.mu+0.5*self.sigma**2)-1),dtype=tf.float64)
        self.mu_sigma_sq = np.exp(self.mu + 0.5 * self.sigma**2) - 1
        
        self.jump_time_max = int(ss.poisson.ppf(0.99999999, self.lam*self.delta_t))
       
    def Forward_SDE(self,t,x,dt,dw,jump):
        x1 = x + jump
        return x1
    
    def Backward_SDE(self,t,x,dt,dw,y,dy_dx,jump):
        y1 = y + jump
        return y1
    
    def Terminal_condition(self,t,x):
        return x

    def Solution(self,t,x,training=False):
        return {
            'y' : x,
            'u' : self.constant*x,
            'dy_dx':tf.ones_like(x),
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
            'y':solution,
            'u':self.constant*x*grad_x, 
            'dy_dx':grad_x
            } 

    def Element_jump(self,num_sample):
        P_sample = ss.poisson.rvs(self.lam*self.delta_t,size=num_sample)
        output = np.zeros((num_sample, self.jump_time_max, self.dim))
        for idx,k in enumerate(P_sample):
            k_create = min(k,self.jump_time_max)
            if k_create != 0:
                output[idx,:k_create,:self.dim] = np.random.normal(self.mu, self.sigma,(k_create,self.dim))
        return output

    def Forward_jump(self,t,x,dt,dw,element_jump):
        element_jump = element_jump.reshape((-1,self.dim))
        Jumps_sample = x * (np.sum((np.exp(element_jump)-1).reshape((-1, self.jump_time_max, self.dim)), axis=1)) - dt * self.lam * self.mu_sigma_sq * x
        return Jumps_sample

    def jump_bate_fun(self,t,x,jump):
        return x*tf.exp(jump)

if __name__ == "__main__":
    # init logging
    logging.basicConfig(
        format = '%(levelname)-6s %(message)s',
        level=logging.INFO
    )

    PDE_name = "oneDimPureJump"
    T = 1
    sample_size = 1000
    num_time_interval = 50
    dim = 1
    model = FeedForwardNet([16,16],tf.nn.relu,output_dim=1)
    mySolver = oneDimPureJump(T,
                            sample_size, num_time_interval, dim,
                            model)
    logging.info(f'Begin to solve {PDE_name}')

    # 结果保存
    path_dir = Tools.init_saving_path(rootDir = "./"+PDE_name,
                           model_config= mySolver.model.get_config())
    logging.info(f'logfile Dir:{path_dir["rootDir"]}')

    # 检查损失函数是否合理
    mySolver.check_loss_on_solution(sample_size = 100)

    # 定义学习率调度器
    boundaries = [4000, 6000, 7000, 8000]
    values = [1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_fn)

    checkpoint = Tools.CheckPoint(model=mySolver.model,dir=path_dir['checkpoint_path'])
    
    # 训练模型
    mySolver.train(num_iterations=1000, optimizer=optimizer)

    # 保存结果
    Tools.dump_model(mySolver.model,path_dir["model_path"])
    Tools.save_trainHistory(mySolver.training_history,path_dir["training_history_file"])