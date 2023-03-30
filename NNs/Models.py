# CCT: Escaping the Big Data Paradigm with Compact Transformers
# Paper: https://arxiv.org/pdf/2104.05704.pdf
# CCT-L/KxT: 
# K transformer encoder layers 
# T-layer convolutional tokenizer with KxK kernel size.
# In their paper, CCT-14/7x2 reached 80.67% Top-1 accruacy with 22.36M params, with 300 training epochs wo extra data
# CCT-14/7x2 also made SOTA 99.76% top-1 for transfer learning to Flowers-102, which makes it a promising candidate for fine-grained classification

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from NNs.Blocks import *
from NNs.Layers import *
from NNs.utils import *


def xpdViT(target_keep_rate, N = 1024, D = 36, num_transformer_layers = 5):
    inputs = keras.Input(shape = (32, 32, 3))
    x = inputs
    x = tf.keras.layers.Conv2D(kernel_size= 3, strides = 1, filters = D, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    P = x.shape[1]
    #print(x.shape)
    x = tf.reshape(x, (-1, P*P, D))
    N_keep_tokens = [N]
    for ii in range(num_transformer_layers):
        N_keep_tokens += [int(-(N*(1-target_keep_rate))/(num_transformer_layers)*(ii+1)+N)]
        N_throw_tokens = N_keep_tokens[ii] - N_keep_tokens[ii+1]
        print(N_keep_tokens, N_throw_tokens)
        att, weight = MultiHeadSelfAttention(num_heads = 6, output_weight = True)(x)
        print(att.shape, weight.shape)
        if target_keep_rate < 1.0:
            att = TokenRemoval(num_discard_tokens = N_throw_tokens,
                            #tot_tokens = N_keep_tokens[ii],
                            num_keep_tokens = N_keep_tokens[ii+1],
                            embedding_dims = D                      
                            )([att, weight])
            N_keep_tokens[ii+1] = N_keep_tokens[ii+1]+1
            att = tf.reshape(att, (-1, N_keep_tokens[ii+1], D))           
    
    x =  tf.keras.layers.LayerNormalization()(att)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(.5)(x)
    outputs = keras.layers.Dense(100, activation = 'softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def ViLin(N = 256, D = 63, K = 256, Dropout_rate = 0.2, DropPath_rate = 0.2, num_transformer_layers = 11):
  inputs = keras.Input(shape = (32, 32, 3))
  x = inputs
  #x = tf.keras.layers.Conv2D(kernel_size= 3, strides = 2, filters = D//2, padding = 'same')(inputs)
  x = tf.keras.layers.Conv2D(kernel_size= 3, strides = 1, filters = D, padding = 'same')(x)
  x = tf.keras.layers.Conv2D(kernel_size= 3, strides = 1, filters = D, padding = 'same')(x)
  x = tf.keras.layers.MaxPooling2D(pool_size = 3, strides = 2)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  P = x.shape[1]
  x = tf.reshape(x, (-1, P*P, D))
  for ii in range(num_transformer_layers):
    att = MultiHeadSelfLinearAttention(num_heads = 7, kv_reprojection_dim= K, kernel_size = 3)(x)
    att = tf.keras.layers.Dropout(Dropout_rate)(att)
    att = DropPath(0.1)(att)
    x = tf.keras.layers.Add()([x, att])
    x =  tf.keras.layers.LayerNormalization()(x)
    ffn = FeedForwardNetwork(mlp_ratio = 7, DropOut_rate = 0.2)(x)
    ffn = DropPath(0.1)(ffn)
    x = tf.keras.layers.Add()([x, ffn])
    x =  tf.keras.layers.LayerNormalization()(x)

  x = SeqPool(n_attn_channel = 75)(x)
  x = tf.keras.layers.Dropout(.5)(x)
  outputs = keras.layers.Dense(100, activation = 'softmax')(x)
  model = tf.keras.Model(inputs, outputs)
  return model