import tensorflow as tf

class FeedForwardNet(tf.keras.Model):
    def __init__(self, num_hiddens, activate_fun, output_dim:int = 1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.output_dim = output_dim
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,
                                                   bias_initializer=tf.initializers.GlorotUniform)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(output_dim, use_bias=True, activation=None,
                                                       bias_initializer=tf.initializers.GlorotUniform))
        self.activate_fun = activate_fun

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> tanh) * len(num_hiddens) -> dense -> bn"""
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.activate_fun(x)
        x = self.dense_layers[-1](x)
        return x
    
    def get_config(self):
        return {
            "num_hiddens":self.num_hiddens,
            "activate_fun":self.activate_fun.__name__,
            "output_dim":self.output_dim
        }
        
    @classmethod
    def from_config(cls, config):
        num_hiddens = config["num_hiddens"]
        activate_fun = getattr(tf.nn, config["activate_fun"])
        output_dim = config['output_dim']
        return cls(num_hiddens=num_hiddens, activate_fun=activate_fun,output_dim = output_dim)