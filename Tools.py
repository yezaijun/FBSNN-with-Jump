import tensorflow as tf
import pandas as pd
import random
import numpy as np
import os
import logging
import datetime
import json

class CheckPoint(tf.train.Checkpoint):
    def __init__(self,dir:str, frequency:int = 500,root=None, **kwargs):
        super().__init__(root, **kwargs)
        self.dir = dir
        self.frequency = frequency

def dump_model(model,filepath):
    logging.info(f'Model dumped in {filepath}')
    # model.save(filepath)
    model.save(filepath, save_format='tf')

def save_trainHistory(history,dir:str):
    df = pd.DataFrame(history,columns=['step','loss','mean_relative_error','t0_relative_error','elapsed_time','learning_rate'])
    df.to_csv(dir,index=False)
    return True

def read_trainHistory(dir:str):
    df = pd.read_csv(dir)
    return df
    
def Set_Seed(random_seed:int=42):
    random.seed(random_seed)  # set random seed for python
    np.random.seed(random_seed)  # set random seed for numpy
    tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu

def custom_rolling_mean(df, column, window_size):  
    """  
    计算滑动平均值，同时去除窗口中的最大值和最小值。  
      
    参数:  
        df (pd.DataFrame): 输入的DataFrame。  
        column (str): 要计算滑动平均的列名。  
        window_size (int): 滑动窗口的大小。  
          
    返回:  
        pd.Series: 包含滑动平均值的Series。  
    """  
    # 初始化结果列表  
    results = []  
      
    # 遍历DataFrame的每一行  
    for i in range(len(df)):  
        # 获取当前窗口的数据  
        window_data = df[column].iloc[max(0, i - window_size + 1):i + 1]  
          
        # 如果窗口中的数据少于窗口大小，则跳过  
        if len(window_data) < window_size:  
            results.append(np.nan)  
            continue  
          
        # 去除窗口中的最大值和最小值，然后计算平均值  
        window_data_filtered = window_data[window_data != window_data.max()]  
        window_data_filtered = window_data_filtered[window_data_filtered != window_data_filtered.max()]  
        results.append(window_data_filtered.mean())  
      
    # 将结果转换为Series并返回  
    return pd.Series(results, index=df.index) 

def init_saving_path(rootDir,pde_config=None,model_config=None):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    now_train_dir = rootDir + '_'+timestamp + '/'
    os.makedirs(now_train_dir)
    if pde_config:
        with open(now_train_dir + "pde_config.json", 'w') as f:
            json.dump(pde_config,f)

    if model_config:
        with open(now_train_dir + "network_config.json", 'w') as f:
            json.dump(model_config,f)


    return {
        "rootDir": now_train_dir,
        "checkpoint_path" : now_train_dir+ "checkpoints/model.ckpt",
        "model_path" : now_train_dir+ "model",
        "training_history_file" : now_train_dir+ "history.csv"
    }

def visual_check(mySolver,sample_size):
    temp = sample_size
    x_initial = mySolver.x_init_generator(temp)
    input_sample = mySolver.create_sample(temp,x_initial)
    dw,dt = input_sample['dw'],input_sample['dt']
    t,x = input_sample['t'],input_sample['x']
    element_jump_sample = input_sample['element_jump']

    Fitter_output = mySolver.Solution(t,x, training=False)
    y = Fitter_output['y']
    dy_dx = Fitter_output['dy_dx']
    u = Fitter_output['u']
    jump = mySolver.Backward_jump(t=t,x=x,dt=dt,dw=dw,y=y,dy_dx=dy_dx,u=u,
                                  element_jump=element_jump_sample,PDE_Fitter=mySolver.Solution,training=False)
    y1 = mySolver.Backward_SDE(t=t,x=x,dt=dt,dw=dw,y=y,dy_dx=dy_dx,jump=jump)  

    y_with_shape = tf.reshape(y,[-1,(mySolver.num_time_interval+1),1])
    y1_with_shape = tf.reshape(y1,[-1,(mySolver.num_time_interval+1),1])
    y_terminal = mySolver.Terminal_condition(
        t= tf.reshape(t,[-1,(mySolver.num_time_interval+1),1])[:,-1,:],
        x= tf.reshape(x,[-1,(mySolver.num_time_interval+1),mySolver.dim])[:,-1,:])
      

    y_total = tf.concat([y_with_shape[:,1:,:],y_with_shape[:,-1:,:]],axis=1)
    y1_total = tf.concat([y1_with_shape[:,:-1,:],tf.reshape(y_terminal,[-1,1,1])],axis=1)

    import matplotlib.pylab as plt
    for i in range(temp):
        plt.plot(y_total[i,:,:],'r-.')
        plt.plot(y_total[i,:,:],'b',alpha=0.5)
    delta = y1_total - y_total
    DELTA_CLIP = 50
    loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta), 2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
    print(loss.numpy())
    plt.show()

    plt.plot(tf.reduce_mean(delta,axis=0))
    plt.show()