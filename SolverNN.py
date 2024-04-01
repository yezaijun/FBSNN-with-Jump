import numpy as np
import tensorflow as tf
import time
import logging

from Network import *
from Tools import *

DELTA_CLIP = 50.0
tf.keras.backend.set_floatx('float64')

class FBSNNJ(object): # Forward-Backward Stochastic Neural Network with Jump
    def __init__(self, terminal_time,
                       sample_size, num_time_interval, dim,
                       model:tf.keras.Model):
        """
        Initialize the Forward-Backward Stochastic Neural Network with Jump (FBSNNJ).

        Args:
        - terminal_time: The terminal time of the simulation.
        - sample_size: Number of trajectories to generate.
        - num_time_interval: Number of time intervals.
        - dim: Number of dimensions.
        - model: The neural network model for approximation.
        """
        
        self.terminal_time = terminal_time # terminal time
        
        self.sample_size = sample_size # number of trajectories
        self.num_time_interval = num_time_interval # number of time snapshots
        self.dim = dim # number of dimensions

        self.delta_t = self.terminal_time/self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        
        self.model = model
        
        self.training_history = []

        self.trained_iterations = 0
        self.trained_time = 0

        self.jump_time_max = 1
        self.x_initial = self.x_init_generator(sample_size)
    
    def get_config(self):
        return {
            "terminal_time" : self.terminal_time,
            "sample_size" : self.sample_size,
            "num_time_interval" : self.num_time_interval,
            "dim" : self.dim
        }

    def train(self,num_iterations:int,
              optimizer:tf.keras.optimizers.Optimizer,
              logging_frequency:int=10,
              logging_verbose:bool=True,
              checkpoint:CheckPoint=None,
              summary_writer = None):
        """
        Train the FBSNNJ model.

        Args:
        - num_iterations: Number of training iterations.
        - optimizer: The optimizer for training.
        - logging_frequency: Frequency of logging.
        - logging_verbose: Whether to log verbose information.
        - checkpoint: Object for managing checkpoints.
        - summary_writer: Object for writing summaries.
        """
        self.optimizer:tf.keras.optimizers.Optimizer = optimizer
        
        start_time = time.time()
        for step in range(num_iterations):
            input_sample = self.create_sample(self.sample_size,x_initial=self.x_initial)
            self.train_step(input_sample)

            if step % logging_frequency == 0:
                self.logging_function(start_time,step,logging_verbose,summary_writer)
            if checkpoint and (step%checkpoint.frequency ==0):
                checkpoint.save(checkpoint.dir)

        self.logging_function(start_time,step+1,logging_verbose)
        
        self.trained_iterations += num_iterations
        self.trained_time += time.time() - start_time

    def loss_function(self,input_sample,PDE_Fitter,training:bool=False):
        """
        Compute the loss function of the FBSNN.

        Args:
        - input_sample: Input sample for loss computation.
        - PDE_Fitter: The model for fitting the PDE.
        - training: Whether the model is in training mode.

        Returns:
        - Loss value.
        """
        dw,dt = input_sample['dw'],input_sample['dt']
        t,x = input_sample['t'],input_sample['x']
        element_jump_sample = input_sample['element_jump']

        Fitter_output = PDE_Fitter(t,x, training)
        y = Fitter_output['y']
        dy_dx = Fitter_output['dy_dx']
        u = Fitter_output['u']

        jump = self.Backward_jump(t=t,x=x,dt=dt,dw=dw,y=y,dy_dx=dy_dx,u=u,
                                  element_jump=element_jump_sample,PDE_Fitter=PDE_Fitter,training=training)
        y1 = self.Backward_SDE(t=t,x=x,dt=dt,dw=dw,y=y,dy_dx=dy_dx,jump=jump)

        y_with_shape = tf.reshape(y,[-1,(self.num_time_interval+1),1])
        y1_with_shape = tf.reshape(y1,[-1,(self.num_time_interval+1),1])
        y_terminal = self.Terminal_condition(
            t= tf.reshape(t,[-1,(self.num_time_interval+1),1])[:,-1,:],
            x= tf.reshape(x,[-1,(self.num_time_interval+1),self.dim])[:,-1,:])
        
        y_total = tf.concat([y_with_shape[:,1:,:],y_with_shape[:,-1:,:]],axis=1)
        y1_total = tf.concat([y1_with_shape[:,:-1,:],tf.reshape(y_terminal,[-1,1,1])],axis=1)

        delta = y1_total - y_total

        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta), 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        return loss
    
    @tf.function           
    def train_step(self,input_sample):
        """
        Perform a single training step.

        Args:
        - input_sample: Input sample for training.
        """
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_function(input_sample,PDE_Fitter=self.model_approximate, training=True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                
    def logging_function(self,start_time:float,step:int,logging_verbose:bool=True,summary_writer=None):
        step = self.trained_iterations + step
        input_sample = self.create_sample(self.sample_size,x_initial=self.x_initial)
        loss = self.loss_function(input_sample,PDE_Fitter=self.model_approximate, training = False).numpy()
        elapsed_time = time.time() - start_time + self.trained_time
        model_relative_error = self.model_relative_error(input_sample)
        mean_relative_error = model_relative_error['mean_relative_error']
        t0_relative_error = model_relative_error['t0_relative_error']
        lr = self.optimizer.learning_rate(step).numpy() if callable(self.optimizer.learning_rate) else self.optimizer.learning_rate.numpy()

        data_list = [step, loss, mean_relative_error,t0_relative_error ,elapsed_time ,lr]
        self.training_history.append(data_list)
        if logging_verbose:
            logging.info("step: %5u,\tLoss: %.4e,\tRelativeError: %.4e,\tt0_RelativeError:%.4e,\telapsed time: %3u,\tLearningRate: %.4e" % (step, loss, mean_relative_error,t0_relative_error ,elapsed_time,lr))
        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=step)
                tf.summary.scalar("mean_relative_error", mean_relative_error, step=step)
                tf.summary.scalar("t0_relative_error", t0_relative_error, step=step)
                tf.summary.scalar("elapsed_time", elapsed_time, step=step)
                tf.summary.scalar("LearningRate", lr, step=step)

        return data_list
    
    def x_init_generator(self,sample_size:int):
        return np.ones((sample_size,self.dim))
    
    def create_sample(self,sample_size:int,x_initial) -> dict:
        dw_sample:np.ndarray = self.sqrt_delta_t * np.random.normal(size=[sample_size,(self.num_time_interval+1), self.dim])
        dt_sample:np.ndarray = self.delta_t * np.ones(shape=(sample_size, (self.num_time_interval+1), 1))
        t_sample:np.ndarray = np.cumsum(dt_sample,axis=1) - self.delta_t
        element_jump_sample:np.ndarray = np.zeros(shape=(sample_size,(self.num_time_interval+1),self.jump_time_max, self.dim))
        x_sample:np.ndarray = np.zeros_like(dw_sample)
        x_sample[:,0,:] = x_initial   
        for i in range(0,self.num_time_interval):
            element_jump_sample[:,i,:,:] = self.Element_jump(sample_size)
            forward_jump = self.Forward_jump(t = t_sample[:,i,:],
                                             x = x_sample[:,i,:],
                                            dt = dt_sample[:,i,:],
                                            dw = dw_sample[:,i,:],
                                  element_jump = element_jump_sample[:,i,:,:])
            x_sample[:,i+1,:] = self.Forward_SDE(t = t_sample[:,i,:],
                                                 x = x_sample[:,i,:],
                                                dt = dt_sample[:,i,:],
                                                dw = dw_sample[:,i,:],
                                              jump = forward_jump)

        return {
            'dw' : dw_sample.reshape([(self.num_time_interval+1)*sample_size,self.dim]),
            'dt' : dt_sample.reshape([(self.num_time_interval+1)*sample_size,1]),
            'x' : x_sample.reshape([(self.num_time_interval+1)*sample_size,self.dim]),
            't' : t_sample.reshape([(self.num_time_interval+1)*sample_size,1]),
            'element_jump' : element_jump_sample.reshape((-1,self.dim))
        }
    
    def model_approximate(self,t,x,training):
        return NotImplementedError

    def Forward_SDE(self,t,x,dt,dw,jump):
        return NotImplementedError
    
    def Backward_SDE(self,t,x,dt,dw,y,dy_dx,jump):
        return NotImplementedError
    
    def Terminal_condition(self,t,x):
        return NotImplementedError
    
    def Solution(self,t,x):
        return NotImplementedError
    
    def Element_jump(self,num_sample):
        return np.zeros((num_sample, self.jump_time_max, self.dim))

    def Forward_jump(self,t,x,dt,dw,element_jump):
        return tf.zeros_like(x)
    
    def jump_bate_fun(self,t,x,jump):
        return x
    
    def Backward_jump(self,t,x,dt,dw,y,dy_dx,u,element_jump,PDE_Fitter,training):
        x_repeated = tf.repeat(x,repeats=self.jump_time_max,axis=0)
        y_repeated = tf.repeat(y,repeats=self.jump_time_max,axis=0)
        t_repeated = tf.repeat(t,repeats=self.jump_time_max,axis=0)

        jump_x = self.jump_bate_fun(t = t_repeated,
                                    x = x_repeated,
                                    jump = element_jump)

        Fitter_output = PDE_Fitter(t_repeated, jump_x, training)
        y_jump = Fitter_output['y']

        jump3D = tf.reduce_sum(tf.reshape(y_jump-y_repeated,(-1,(self.num_time_interval+1),self.jump_time_max,1)),axis=2)
        jump2D = tf.reshape(jump3D,(-1,1))

        Jumps_sample = jump2D - dt*u
        return Jumps_sample

    def load_model(self,filepath):
        self.model = tf.keras.models.load_model(filepath,custom_objects={"FeedForwardNet": FeedForwardNet})
        self.model = tf.keras.models.load_model(filepath)
        logging.info(f"Success load Model from {filepath}")

    def model_relative_error(self,input_sample):
        x = input_sample['x']
        t = input_sample['t']
        y = self.model_approximate(t,x)['y']
        u = self.Solution(t,x)['y']

        y0 = tf.reshape(y,(-1,(self.num_time_interval+1),1))[:,0,:]
        u0 = tf.reshape(u,(-1,(self.num_time_interval+1),1))[:,0,:]
        return {
            'mean_relative_error': tf.reduce_mean(abs((y-u)/u)).numpy(),
            't0_relative_error': tf.reduce_mean(abs((y0-u0)/u0)).numpy()
        } 
    
    def check_loss_on_solution(self,sample_size = 100,verbose:bool=True):
        x_initial = self.x_init_generator(sample_size)
        input_sample = self.create_sample(sample_size,x_initial)
        loss = self.loss_function(input_sample,self.Solution).numpy()
        if verbose:
            logging.info(f"loss value on self.Solution:{loss}")
        return loss