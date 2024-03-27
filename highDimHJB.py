import numpy as np
import tensorflow as tf
import logging
import scipy.stats as ss

import Tools 
from Network import *
from SolverNN import FBSNN

class highDimHJB(FBSNN):
    def __init__(self, T, sample_size, num_time_interval, dim, model):
        super().__init__(T, sample_size, num_time_interval, dim, model)
        self.lam = 0.3
        self.theta = np.sqrt(2.0)
        
        self.dim_tf = tf.cast(self.dim,dtype=tf.float64)

        self.jump_time_max = int(ss.poisson.ppf(0.99999999, self.lam*self.delta_t))
 
    def Forward_SDE(self,t,x,dt,dw,jump):
        x1 = x +  self.theta * dw
        return x1
    
    def Backward_jump(self,t,x,dt,dw,y,dy_dx,u,element_jump,PDE_Fitter,training):
        return tf.zeros_like(dt)

    def Backward_SDE(self,t,x,dt,dw,y,dy_dx,jump):
        dy_dx_L2 = tf.reduce_sum(dy_dx**2, axis=1, keepdims=True)
        y1 = y + dy_dx_L2 * dt + self.theta * tf.reduce_sum(dy_dx * dw,axis=1,keepdims=True)
        return y1
    
    def Terminal_condition(self,t,x):
        return tf.math.log(0.5+0.5*tf.reduce_sum(x**2,axis=1,keepdims=True))
    
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
            "u" : tf.zeros_like(grad_x),
            'dy_dx':grad_x
            } 

    def Element_jump(self,num_sample):
        return np.zeros((num_sample, self.jump_time_max, self.dim))

    def Forward_jump(self,t,x,dt,dw,element_jump):
        return tf.zeros_like(dt)
    
    def jump_bate_fun(self,t,x,jump):
        return tf.zeros_like(x)
    

    def model_relative_error(self,input_sample):
        return {
            'mean_relative_error': np.nan,
            't0_relative_error': np.nan
        } 
    
    def x_init_generator(self,sample_size:int):
        return np.zeros((sample_size,self.dim))

if __name__ == "__main__":
    # init logging
    logging.basicConfig(
        format = '%(levelname)-6s %(message)s',
        level=logging.INFO
    )

    PDE_name = "HighDimHJB"
    T = 1
    sample_size = 500
    num_time_interval = 50
    dim = 1

    model = FeedForwardNet([16,16,16,16],tf.nn.sigmoid,output_dim=1)
    mySolver = highDimHJB(T, sample_size, num_time_interval, dim, model)
    logging.info(f'Begin to solve {PDE_name}')

    # 检查损失函数是否合理
    # mySolver.check_loss_on_solution(sample_size = 1000)

    # 结果保存
    path_dir = Tools.init_saving_path(rootDir = "./"+PDE_name,
                                      pde_config = mySolver.get_config(),
                                      model_config= mySolver.model.get_config())
    logging.info(f'logfile Dir:{path_dir["rootDir"]}')

    # 定义学习率调度器
    boundaries = [1000,6000]
    values = [1e-2, 1e-3,1e-6]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_fn)

    checkpoint = Tools.CheckPoint(model=mySolver.model,dir=path_dir['checkpoint_path'])
    
    # 训练模型
    mySolver.train(num_iterations=20,optimizer=optimizer)

    # # # 保存结果
    Tools.dump_model(mySolver.model,path_dir["model_path"])
    Tools.save_trainHistory(mySolver.training_history,path_dir["training_history_file"])