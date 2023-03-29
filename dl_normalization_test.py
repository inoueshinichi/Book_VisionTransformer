from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import datetime

def test_name_deco(f):
    def _wrapper(*args, **kargs):

        print(f"{datetime.datetime.now()} [Start] {f.__name__}")

        v = f(*args, **kargs)

        print(f"{datetime.datetime.now()} [End] {f.__name__}")

        return v
    
    return _wrapper
        

@test_name_deco
def test_batch_norm_1d():
    # Must Input : (N, C) or (N, C, L)
    N = 2
    C = 3
    L = 4

    print('(N, C) input [L=1]--------------------------------------')

    # (N, C)
    input_2dim = torch.ones(N, C)
    print('input_2dim.shape', input_2dim.shape)
    print('input_2dim: \n', input_2dim)

    batch_norm_1d = nn.BatchNorm1d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定
    output_2dim = batch_norm_1d(input_2dim)
    print('output_2dim.shape', output_2dim.shape)
    print('output_2dim: \n', output_2dim)

    gamma = batch_norm_1d.weight
    beta = batch_norm_1d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

    print('(N, C, L) input [Main]--------------------------------------')

    # (N, C, L)
    input_3dim = torch.ones(N, C, L)
    print('input_3dim.shape', input_3dim.shape)
    print('input_3dim: \n', input_3dim)

    batch_norm_1d = nn.BatchNorm1d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定
    output_3dim = batch_norm_1d(input_3dim)
    print('output_3dim.shape', output_3dim.shape)
    print('output_3dim: \n', output_3dim)

    gamma = batch_norm_1d.weight
    beta = batch_norm_1d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

@test_name_deco
def test_batch_norm_2d():
    # Must Input : (N, C, H, W)
    N = 2
    C = 3
    H = 4
    W = 4

    batch_norm_2d = nn.BatchNorm2d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定

    print('(N, C, H, W) input [Main]--------------------------------------')

    input_4dim = torch.ones(N, C, H, W)
    print('input_4dim.shape', input_4dim.shape)
    print('input_4dim: \n', input_4dim)

    output_4dim = batch_norm_2d(input_4dim)
    print('output_4dim.shape', output_4dim.shape)
    print('output_4dim: \n', output_4dim)

    gamma = batch_norm_2d.weight
    beta = batch_norm_2d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)


@test_name_deco
def test_batch_norm_3d():
    # Must Input : (N, C, D, H, W)
    N = 2
    C = 3
    D = 4
    H = 4
    W = 4

    batch_norm_3d = nn.BatchNorm3d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定

    print('(N, C, D, H, W) input [Main]--------------------------------------')





@test_name_deco
def test_layer_norm():
    # Input : (N, C, H, W), (N, C, D), etc

    N = 2
    C = 3
    H = 4
    W = 4
    input = torch.ones(N,C,H,W)
    print('input.shape', input.shape)
    print('input: \n', input)

    output = input.mean((-2, -1))
    print('output.shape', output.shape)
    print('output: \n', output)

    layer_norm = nn.LayerNorm(normalized_shape=(H,W), elementwise_affine=True)
    output_layer_norm = layer_norm(input)
    print('output_layer_nrom.shape', output_layer_norm.shape)
    print('output_layer_nrom', output_layer_norm)

    gamma = layer_norm.weight # γ (スケール学習パラメータ)
    beta = layer_norm.bias # β (シフト学習パラメータ)
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape',beta.shape)
    print('beta: \n', beta)
    

if __name__ == "__main__":
    # Batch Norm
    # test_batch_norm_1d()
    test_batch_norm_2d()

    # Layer Norm
    # test_layer_norm()
