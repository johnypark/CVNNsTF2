import tensorflow as tf
from tensorflow import keras
from functools import partial


KERNEL_INIT = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    }} #from resnet-rs
    
def BasicConv2D(filters,
              kernel_size,
              strides = 1,
              padding = "same",
              activation: str = "relu",
              use_bias = False,
              kernel_initializer = {
                "class_name": "VarianceScaling",
                "config": {"scale": 2.0, "mode": "fan_out",
                           "distribution": "truncated_normal" }}, 
              bn_momentum = 0.0,
              bn_epsilon = 1e-5,
              name = None, 
              **kwargs):
    """ 
    
    ConvBlock: Base unit of ResNet. keras.layers.Conv2D + BN + activation layers.

    Args: Argument style inherits keras.layers.Conv2D
        filters (int): # of channels.
        kernel_size (int): kernel size.
        strides (int, optional): strides in the Conv2D operation . Defaults to 1.
        padding (str, optional): padding in the Conv2D operation. Defaults to "same".
        activation (str, optional): name of the activation function. keras.layers.Activation. Defaults to "relu".
        name (str, optional): name of the layer. Defaults to None.
        
    """
    if name is None: # adopted this structure from tf.kera
        counter = keras.backend.get_uid("conv_")
        name = f"conv_{counter}"
      
    def apply(inputs):
        x = inputs
        x = keras.layers.Conv2D(filters = filters,
                                kernel_size = kernel_size,
                                padding = padding,
                                strides = strides,
                                name = name + "_{}x{}conv_ch{}".format(
                                    kernel_size, kernel_size, filters),
                                kernel_initializer = kernel_initializer,
                                use_bias = use_bias,
                                **kwargs
                                )(x)
        x = keras.layers.BatchNormalization( momentum = bn_momentum,
                                             epsilon = bn_epsilon,
            name = name +"_batch_norm")(x)
        if activation:
            x = keras.layers.Activation(activation, name = name +"_act")(x)
        return x
    
    return apply

# MHSA layer 
# Adopted from: https://github.com/faustomorales/vit-keras/blob/master/vit_keras/utils.py
# Also learn: https://keras.io/guides/making_new_layers_and_models_via_subclassing/

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, *args, num_heads, DropOut_rate = 0.1, output_weight = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.output_weight = output_weight
        self.DropOut_rate = DropOut_rate

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads !=0:
          raise ValueError(
              f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
              )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = keras.layers.Dense(hidden_size, name = "dense_query")
        self.key_dense = keras.layers.Dense(hidden_size, name = "dense_key")
        self.value_dense = keras.layers.Dense(hidden_size, name = "dense_value")
        self.out_dense = keras.layers.Dense(hidden_size, name = "dense_out")
        self.Dropout = keras.layers.Dropout(rate = self.DropOut_rate)
        self.CalcAttention = partial(self.ScaledDotProductAttention, dim  = self.projection_dim)

    def ScaledDotProductAttention(self, query, key, value, dim):
        score = tf.matmul(query, key, transpose_b = True)
        #dim_key = tf.cast(tf.shape(key)[-1], dtype = score.dtype)
        scaled_score = score / tf.math.sqrt(dim)
        weights = tf.nn.softmax(scaled_score, axis = -1)
        weights = self.Dropout(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_to_multihead(self, x, batch_size):
        x = tf.reshape(
                      tensor = x, 
                      shape = (batch_size, -1, self.num_heads, self.projection_dim)
                      )
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query, key, value = [self.separate_to_multihead(tensor, batch_size) for tensor in [query, key, value]]
        
        weighted_value, weights = self.CalcAttention(query, key, value)
        weighted_value = tf.transpose(weighted_value, perm = [0, 2, 1, 3])
        combined_values = tf.reshape(weighted_value, 
                                      shape = (batch_size, -1, self.hidden_size)
                                      )
        output = self.out_dense(combined_values)
        output = self.Dropout(output)
        
        if self.output_weight:
            output = output, weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"DropOut_rate": self.DropOut_rate})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Positional embedding
class AddPosEmbedding(keras.layers.Layer):
    
    def __init__(self, 
                 num_patches, 
                 embedding_dim,
                 embedding_type = 'learnable',
                 noise_stddev = 2e-1,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.embedding_type = embedding_type
        self.num_patches = num_patches # this can be removed
        self.embedding_dim = embedding_dim # this can be removed
        self.noise_stddev = noise_stddev
    
    
    def sinusodial_embedding(self, num_patches, embedding_dim):
    
        """ sinusodial embedding in the attention is all you need paper 
        example:
        >> plt.imshow(sinusodial_embedding(100,120).numpy()[0], cmap='hot',aspect='auto')
        """
    
        def criss_cross(k):
            n_even = k - k//2
            even = list(range(n_even))
            odd = list(range(n_even, k))
            ccl = []
            for i in range(k//2):
                ccl = ccl+ [even[i]]+ [odd[i]]
            if k//2 != k/2:
                ccl = ccl + [even[k//2]]
            return ccl
            
        embed = tf.cast(([[p / (10000 ** (2 * (i//2) / embedding_dim)) for i in range(embedding_dim)] for p in range(num_patches)]), tf.float32)
        even_col =  tf.sin(embed[:, 0::2])
        odd_col = tf.cos(embed[:, 1::2])
        embed = tf.concat([even_col, odd_col], axis = 1)
        embed = tf.gather(embed, criss_cross(embedding_dim), axis = 1)
        embed = tf.expand_dims(embed, axis=0)

        return embed
    
    def build(self, input_shape):
        assert (
            len(input_shape)==3 
        ), "Expected tensor dim=3. Got {}".format(len(input_shape))
            
        num_patches = input_shape[-2]
        embedding_dim = input_shape[-1]
        if self.embedding_type:
            if self.embedding_type == 'sinusodial':
                self.positional_embedding = tf.Variable(self.sinusodial_embedding(num_patches = num_patches,
                                                embedding_dim = embedding_dim,
                                                name ='sinosodial'
                                                ),
                        trainable = False)
            elif self.embedding_type == 'learnable':
                    self.positional_embedding = tf.Variable(
                        tf.random.truncated_normal(shape=[1, num_patches, embedding_dim], stddev= self.noise_stddev),
                        trainable = True,
                        name = 'learnable')
                
        else: # else simple gaussian noise injection
                
            noise = tf.random_normal_initializer(stddev = self.noise_stddev) 
            self.positional_embedding = tf.Variable(
                    noise(shape = [1, num_patches, embedding_dim]),
                    trainable = False,
                    name = 'gaussian_noise')
            
    def call(self, input):
        PE = tf.cast(self.positional_embedding, dtype = input.dtype)
        input = tf.math.add(input, PE)
        return input
    
    
class MultiHeadSelfLinearAttention(keras.layers.Layer):
    def __init__(self, *args, 
                 num_heads, 
                 kv_reprojection_dim,
                 attention_DropOut_rate = 0.1,
                 DropOut_rate = 0.1, 
                 kernel_size = None,
                 output_weight = False, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.k = kv_reprojection_dim
        self.DropOut_rate = DropOut_rate
        self.attention_DropOut_rate = attention_DropOut_rate
        self.output_weight = output_weight
        self.kernel_size = kernel_size
 
    def build(self, input_shape):
        batch_size = input_shape[0]
        embedding_dim = input_shape[-1]
        num_tokens = input_shape[-2]
        num_heads = self.num_heads
        if embedding_dim % num_heads !=0:
          raise ValueError(
              f"embedding dimension = {embedding_dim} should be divisible by number of heads = {num_heads}"
              )
        self.b = batch_size
        self.d = embedding_dim
        self.n = num_tokens 
        self.d_sep = self.d // num_heads
        self._query_dense = keras.layers.Dense(self.d, name = "query")
        self._kv_dense = keras.layers.Dense(self.d*2, name = "key_and_value")
        #self._token_reproj = keras.layers.EinsumDense("bnd,nk->bkd", 
        #                                          output_shape = (self.k, None),
        #                                          name = "token_reproj"
        #                                          )
        if self.kernel_size:
            self._channelwise_conv1D = keras.layers.DepthwiseConv1D(kernel_size = self.kernel_size, 
                                                                    strides = self.kernel_size,
                                                                    activation = None)
        self._out_dense = keras.layers.Dense(self.d, name = "out")
        self._Dropout = keras.layers.Dropout(rate = self.DropOut_rate)

    def sep_heads(self, x, num_heads):
        b = self.b
        h = num_heads
        d = x.shape[-1]
        n = x.shape[-2]
        x = tf.reshape(
                      tensor = x, 
                      shape = (-1, n, h, d//h)
                      )
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key):
        score = tf.einsum("bhnd, bhkd -> bhnk", query, key)
        scale = tf.math.sqrt(tf.cast(tf.shape(key)[-1], dtype = score.dtype))
        scaled_score = score / scale
        weights = tf.nn.softmax(scaled_score, axis = -1)
        weights = keras.layers.Dropout(self.attention_DropOut_rate)(weights)
        return weights

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self._query_dense(inputs)
        kv = self._kv_dense(inputs) 
        if self.kernel_size:
            kv = self._channelwise_conv1D(kv)
        kv_reproj = kv #self._token_reproj(kv)
        kv_sep = self.sep_heads(kv_reproj, num_heads = self.num_heads)
        query_sep = self.sep_heads(query, num_heads = self.num_heads)
        key_sep, val_sep = tf.split(kv_sep, 2, axis = -1)
        attention_weight = self.scaled_dot_product_attention(query_sep, key_sep)
        attention_value = attention_weight@val_sep
        combined_values = tf.reshape(attention_value, shape = (-1, self.n, self.d))
        output = self._out_dense(combined_values)
        output = self._Dropout(output)

        if self.output_weight:
            output = output, tf.reduce_mean(attention_weight, 1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads,
                       "kv_reprojection_dim": self.k})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# Sequence Pooling with additional channel option
def SeqPool(n_attn_channel = 1): 
    """ Learnable pooling layer. Replaces the class token in ViT.
    In the paper they tested static pooling methods but learnable weighting is more effcient, 
    because each embedded patch does not contain the same amount of entropy. 
    Enables the model to apply weights to tokens with repsect to the relevance of their information
    """
    def apply(inputs):
        x = inputs    
        x_init = x
        x = tf.keras.layers.Dense(units = n_attn_channel, activation = 'softmax')(x)
        w_x = tf.matmul(x, x_init, transpose_a = True)
        w_x = tf.keras.layers.Flatten()(w_x)     
        return w_x

    return apply


class TokenRemoval(keras.layers.Layer):
    """ 
    TokenRemoval Layer. Adopted idea from https://arxiv.org/abs/2202.07800.
    """
    def __init__(self, 
                 num_discard_tokens, 
                 #tot_tokens, 
                 embedding_dims,
                 num_keep_tokens = None, random = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_keep_tokens = num_keep_tokens
        self.num_discard_tokens = num_discard_tokens
        #self.tot_tokens = tot_tokens
        self.embedding_dims = embedding_dims
        self.random = random

    def invert_index(self, batch_index, max_range):
        """
        inverting index number given maximum range. 
        Does not work for ragged tensor.
        """
        #assert tf.math.reduce_max(batch_index) < max_range, "max_range must be bigger than the maximum index within given tensor."
        compare = tf.expand_dims(batch_index, 2) - tf.expand_dims(tf.range(max_range),0)
        merge_seq_axis = tf.math.reduce_prod(compare, 1)
        batch_index = tf.transpose(tf.where(merge_seq_axis))
        batch_info = tf.gather_nd(batch_index, [[0]])
        full_seq = tf.gather_nd(batch_index, [[1]])
        n_batch = tf.reduce_max(batch_info) + 1
        n_batch = tf.cast(n_batch, tf.int32)

        return tf.reshape(
                full_seq, (n_batch, -1))    
        
    @tf.function(jit_compile=True)
    def get_index_to_keep(self, token_score, num_keep_tokens):
        #print(num_keep_tokens)
        out = tf.math.approx_max_k(token_score, num_keep_tokens)
        return out[1] #tf.math.top_k(token_score, k = num_keep_tokens, sorted = False)[1]# 

    @tf.function(jit_compile=True)
    def get_index_to_discard(self, token_score, num_discard_tokens):
        #print(num_keep_tokens)
        out = tf.math.approx_min_k(token_score, num_discard_tokens)
        idxs_discard = out[1]
        return idxs_discard #tf.math.top_k(token_score, k = num_keep_tokens, sorted = False)[1]# 
    
    @tf.function(jit_compile=True)
    def average_over_Q(self, attention_map):
        return tf.einsum('bnk -> bn', attention_map)

    def build(self, input_shape):
        self.map_shape = input_shape[1]
        self.map_shape = input_shape[0]
        

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("input must be a list with input tensor and corresponding attention map.")
        x, atten_map = inputs 
        token_score = self.average_over_Q(atten_map)
        keep_idx  = self.get_index_to_keep(token_score, num_keep_tokens = self.num_keep_tokens)
        discard_idx = self.get_index_to_discard(token_score, num_discard_tokens = self.num_discard_tokens)
            
        attentive_tokens = tf.gather(x, keep_idx, axis = 1, batch_dims = 1)
        inattentive_tokens = tf.gather(x, discard_idx, axis =1 ,batch_dims =1)
        fused_tokens = tf.keras.layers.GlobalAveragePooling1D(keepdims = True)(inattentive_tokens)
        #out_num_tokens = self.tot_tokens - self.num_discard_tokens
        #out_shape = (-1, out_num_tokens, self.embedding_dims)
        #print("output_shape:{}".format(out_shape))
        return tf.concat([attentive_tokens, fused_tokens], axis =1)#tf.reshape(out, out_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)