import tensorflow as tf
from tensorflow import keras
from NNs.utils import DropPath
from functools import partial
from NNs.Layers import *

# Feed Forward Network (FFN)

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, *args, mlp_ratio,
                                DropOut_rate,
                                activation = 'gelu',
                                **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_ratio = mlp_ratio
        self.DropOut_rate = DropOut_rate
        self.activation = activation
    
    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        overhead_dim = int(embedding_dim*self.mlp_ratio)
        self.Dense_hidden = tf.keras.layers.Dense(units = overhead_dim, name = "dense_hidden")
        self.Dense_out = tf.keras.layers.Dense(units = embedding_dim, name = "dense_out")
        self.Activation = tf.keras.layers.Activation(self.activation)
        self.Dropout = tf.keras.layers.Dropout(rate = self.DropOut_rate)
        
    
    def call(self, inputs):
        x = inputs
        x = self.Dense_hidden(x)
        x = self.Activation(x)
        x = self.Dropout(x)
        x = self.Dense_out(x)
        x = self.Activation(x)
        outputs = self.Dropout(x)
        
        return outputs
    
    
    def get_config(self):
        config = super().get_config()
        config.update({"mlp_ratio": self.mlp_ratio})
        config.update({"DropOut_rate": self.DropOut_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Transformer Block

def Transformer_Block(mlp_ratio,
                      num_heads,
                      stochastic_depth_rate = None,
                      DropOut_rate = 0.1,
                      LayerNormEpsilon = 1e-6):
    def apply(inputs):
        
        x = inputs
        #Attention
        LN_output1 = tf.keras.layers.LayerNormalization(
			epsilon = LayerNormEpsilon
		    )(inputs)
        att = MultiHeadSelfAttention(
			num_heads = num_heads,
            DropOut_rate = DropOut_rate
			)(LN_output1)
        if stochastic_depth_rate:
            att = DropPath(stochastic_depth_rate)(att)
        att_output = tf.keras.layers.Add()([x, att])
        
        #Feed Forward Network
        x1 = att_output
        LN_output2 = tf.keras.layers.LayerNormalization(
            epsilon = LayerNormEpsilon
            )(att_output)
        mlp = FeedForwardNetwork(mlp_ratio = mlp_ratio,
                      DropOut_rate = DropOut_rate 
		    )(LN_output2)
        if stochastic_depth_rate:
            mlp = DropPath(stochastic_depth_rate)(mlp)
        output = tf.keras.layers.Add()([x1, mlp]) 
                      
        return output
    
    return apply


def MB4D_Block(mlp_ratio,
               embedding_dims,
                      stochastic_depth_rate = None,
                      DropOut_rate = 0.1,
                      activation = 'gelu'):
    def apply(inputs):
        
        x = inputs
        #poolformer layer
        pooling = tf.keras.layers.AveragePooling2D(pool_size =3, 
                                                   strides = 2)(x)
        pooling_output = tf.keras.layers.Add()([inputs, pooling])
        
        #MLP substitude
        x1 = pooling_output
        x1 = tf.keras.layers.Conv2D(
            activation = None,
            filters = embedding_dims*mlp_ratio,
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = tf.keras.layers.BatchNoramlization()(x1)
        x1 = tf.keras.layers.Dropout(DropOut_rate)(x1)
        x1 = tf.keras.layers.Activation(activation)(x1)
        x1 = tf.keras.layers.Conv2D(
            activation = None,
            filters = embedding_dims,
            kernel_size = 1,
            strides = 1,
            padding = 'same')(x1)
        x1 = tf.keras.layers.BatchNoramlization()(x1)
        x1 = tf.keras.layers.Dropout(DropOut_rate)(x1)
        if stochastic_depth_rate:
            x1 = DropPath(stochastic_depth_rate)(x1)
        
        output = tf.keras.layers.Add()([pooling_output, x1])     
        return output
    
    return apply



def BN_Res_Block( target_channels,
                  BottleNeck_channels, 
                 ResNetType = "C",
                 padding = "same",
                 downsampling = False,
                 activation: str = "relu",
                 name = None):
    
    """
    BN_Res_Block: BottleNeck Residual Block. type A, B, and D
    """
    #if target_channels == BottleNeck_channels:
    #if name is None: # adopted this structure from tf.keras
    #    counter = keras.backend.get_uid("Residual_")
    #    name = f"Residual_{counter}"
    #else:  
    if name is None: # adopted this structure from tf.keras
        counter = keras.backend.get_uid("BN_Residual_")
        name = f"BN_Residual_{counter}"
    
    def apply(inputs):
        prev_channels = inputs.shape[-1]
    #print(inputs.shape[-1])
        r = inputs # r for residual
        skip_connection = inputs 
        DownSamplingStride = 1
        #if target_channels == BottleNeck_channels:
      
        if prev_channels != target_channels:
            DownSamplingStride = 2
            skip_connection = BasicConv2D(filters = target_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = DownSamplingStride,
                              name = name + "_4",
                              activation = None                           
                              )(skip_connection)
    
        r = BasicConv2D(filters = BottleNeck_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = 1,
                              name = name + "_1"                           
                              )(r)
        r = BasicConv2D(filters = BottleNeck_channels,
                              kernel_size = 3,
                              padding = padding,
                              strides = DownSamplingStride,
                              name = name + "_2"                           
                              )(r)
        r = BasicConv2D(filters = target_channels,
                              kernel_size = 1,
                              padding = padding,
                              strides = 1,
                              name = name + "_3",
                              activation = None                           
                              )(r)

        x = tf.keras.layers.Add()([skip_connection, r])
        x = tf.keras.layers.Activation(activation, name = name +"_act")(x)
    
        return x

    return apply


def Inverted_BN_Block(in_channels, 
                      out_channels, 
                      expansion_factor, 
                      stride, 
                      linear = True,
                      use_se=True, 
                      se_ratio=12,
                      **kwargs):
  
    #use_shortcut = stride == 1 and in_channels <= channels
    in_channels = in_channels
    if linear:
        act_ftn = None
    else:
        act_ftn= 'relu6'
    def apply(inputs):
          x = inputs
          skip_connection = inputs
          if expansion_factor != 1:
              expand = in_channels * expansion_factor
              x = BasicConv2D(filters = expand, 
                        kernel_size = 1, 
                        stride = 1,
                        activation = "silu")(x)
          else:
              expand = in_channels

          x = BasicConv2D(filters = expand,
                        kernel_size = 3,
                        stride = stride,
                        groups = expand,
                        activation = act_ftn, **kwargs)(x) # Double check the padding here!
                        # what is pytorch padding = 1 for keras???
                        # Look for padding = 1 in torch!! documents!
          #if use_se:
              # implement SE layer 
          x = tf.keras.layers.Activation(act_ftn)(x)
          x = BasicConv2D(filters = out_channels,
                        kernel_size = 1,
                        strides = 1,
                        activation = act_ftn)(x) 
          x = tf.keras.layers.Add()([skip_connection, x])
          # add activation layer here? 
          return x
    return apply


def SqueezeBlock(channels,
                 activaiton = 'relu'):
    def apply(inputs):
        x = inputs
        skip_connection = inputs
        x = BasicConv2D(channels//2, kernel_size=1,
                  activation = activaiton)(x)
        x = BasicConv2D(channels//4, kernel_size=1,
                  activation = activaiton)(x)
        x = BasicConv2D(channels//2, kernel_size= (3,1),
                  activation = activaiton)(x)
        x = BasicConv2D(channels//2, kernel_size= (1,3),
                  activation = activaiton)(x)
        x = BasicConv2D(channels, kernel_size= (1,1),
                  activation = activaiton)(x)
        output = tf.keras.layers.Add()([skip_connection, x])
        
        return output
    
    return apply
