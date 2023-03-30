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
    
    gamma = batch_norm_1d.weight
    beta = batch_norm_1d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

    output_2dim = batch_norm_1d(input_2dim)
    print('output_2dim.shape', output_2dim.shape)
    print('output_2dim: \n', output_2dim)

    print('(N, C, L) input [Main]--------------------------------------')

    # (N, C, L)
    input_3dim = torch.ones(N, C, L)
    print('input_3dim.shape', input_3dim.shape)
    print('input_3dim: \n', input_3dim)

    batch_norm_1d = nn.BatchNorm1d(num_features=C) # 平均値と標準偏差をベクトルとして集約したい次元を指定
    
    gamma = batch_norm_1d.weight
    beta = batch_norm_1d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

    output_3dim = batch_norm_1d(input_3dim)
    print('output_3dim.shape', output_3dim.shape)
    print('output_3dim: \n', output_3dim)

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

    gamma = batch_norm_2d.weight
    beta = batch_norm_2d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

    print('output_4dim.shape', output_4dim.shape)
    print('output_4dim: \n', output_4dim)


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

    input_5dim = torch.ones(N, C, D, H, W)
    print('input_5dim.shape', input_5dim.shape)
    print('input_5dim: \n', input_5dim)


    gamma = batch_norm_3d.weight
    beta = batch_norm_3d.bias
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape', beta.shape)
    print('beta: \n', beta)

    output_5dim = batch_norm_3d(input_5dim)
    print('output_5dim.shape', output_5dim.shape)
    print('output_5dim: \n', output_5dim)


@test_name_deco
def test_layer_norm():
    # Input : (N, C, H, W), (N, C, D), etc = (N, *)

    v_t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print('v_t.shape', v_t.shape)
    print('v_t: \n', v_t)
    print('v_t.mean((-2,-1))', v_t.mean((-2,-1)))

    N = 2
    C = 3
    H = 2
    W = 2
    input = torch.ones(N,C,H,W)
    print('input.shape', input.shape)
    print('input: \n', input)

    output = input.mean((-2, -1))
    print('output.shape', output.shape)
    print('output: \n', output)

    # 次元(H=2,W=2)の4つの要素からスカラの平均値と標準偏差を計算して,
    # (H=2,W=2)のテンソルに一様にAffine変換を行う(if elementwise_affine=True).
    # Affine変換時のスケールパラメータγとシフトパラメータβの形状は(H=2,W=2).
    layer_norm = nn.LayerNorm(normalized_shape=(H,W), elementwise_affine=True)
    
    gamma = layer_norm.weight # γ (スケールパラメータ with leanable)
    beta = layer_norm.bias # β (シフトパラメータ with learnable)
    print('gamma.shape', gamma.shape)
    print('gamma: \n', gamma)
    print('beta.shape',beta.shape)
    print('beta: \n', beta)

    output_layer_norm = layer_norm(input)
    print('output_layer_nrom.shape', output_layer_norm.shape)
    print('output_layer_nrom: \n', output_layer_norm)

@test_name_deco
def test_instance_norm_1d():
    # Must Input : (N, C, L) or (C, L)

    N = 2
    C = 3
    L = 4

    print('(C, L) input --------------------------------------')

    # (C, L)
    input_2dim = torch.ones(C, L)
    print('input_2dim.shape', input_2dim.shape)
    print('input_2dim: \n', input_2dim)

    affine = True
    instance_norm_1d = nn.InstanceNorm1d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_1d.weight
        beta = instance_norm_1d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)


    output_2dim = instance_norm_1d(input_2dim)
    print('output_2dim.shape', output_2dim.shape)
    print('output_2dim: \n', output_2dim)

    print('(N, C, L) input [Main]--------------------------------------')

    # (N, C, L)
    input_3dim = torch.ones(N, C, L)
    print('input_3dim.shape', input_3dim.shape)
    print('input_3dim: \n', input_3dim)

    affine = False
    instance_norm_1d = nn.InstanceNorm1d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_1d.weight
        beta = instance_norm_1d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

    output_3dim = instance_norm_1d(input_3dim)
    print('output_3dim.shape', output_3dim.shape)
    print('output_3dim: \n', output_3dim)

@test_name_deco
def test_instance_norm_2d():
    # Must Input : (N, C, H, W) or (C, H, W)
    N = 2
    C = 3
    H = 4
    W = 4

    print('(C, H, W) input --------------------------------------')

    input_3dim = torch.ones(C, H, W)
    print('input_3dim.shape', input_3dim.shape)
    print('input_3dim: \n', input_3dim)

    affine = True
    instance_norm_2d = nn.InstanceNorm2d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_2d.weight
        beta = instance_norm_2d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

    output_3dim = instance_norm_2d(input_3dim)
    print('output_3dim.shape', output_3dim.shape)
    print('output_3dim: \n', output_3dim)

    print('(N, C, H, W) input [Main]----------------------------------')

    # (N, C, H, W)
    input_4dim = torch.ones(N, C, H, W)
    print('input_4dim.shape', input_4dim.shape)
    print('input_4dim: \n', input_4dim)

    affine = False
    instance_norm_2d = nn.InstanceNorm2d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_2d.weight
        beta = instance_norm_2d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

    output_4dim = instance_norm_2d(input_4dim)
    print('output_4dim.shape', output_4dim.shape)
    print('output_4dim: \n', output_4dim)

@test_name_deco
def test_instance_norm_3d():
    # Must Input : (N, C, D, H, W) or (C, D, H, W)
    N = 2
    C = 3
    D = 4
    H = 4
    W = 4

    print('(C, D, H, W) input [Main]----------------------------------')
    input_4dim = torch.ones(C, D, H, W)
    print('input_4dim.shape', input_4dim.shape)
    print('input_4dim: \n', input_4dim)

    affine = True
    instance_norm_3d = nn.InstanceNorm3d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_3d.weight
        beta = instance_norm_3d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

    output_4dim = instance_norm_3d(input_4dim)
    print('output_4dim.shape', output_4dim.shape)
    print('output_4dim: \n', output_4dim)


    print('(N, C, D, H, W) input [Main]--------------------------------------')

    input_5dim = torch.ones(N, C, D, H, W)
    print('input_5dim.shape', input_5dim.shape)
    print('input_5dim: \n', input_5dim)

    affine = False
    instance_norm_3d = nn.InstanceNorm3d(num_features=C, affine=affine)

    if affine:
        gamma = instance_norm_3d.weight
        beta = instance_norm_3d.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)

    output_5dim = instance_norm_3d(input_5dim)
    print('output_5dim.shape', output_5dim.shape)
    print('output_5dim: \n', output_5dim)

@test_name_deco
def test_group_norm():
    # Must Input : (N, C, *)
    N = 2
    C = 9
    H = 2
    W = 2

    print('(N, C, H, W) input ------------------------------------')

    input_4dim = torch.ones(N, C, H, W)
    print('input_4dim.shape', input_4dim.shape)
    print('input_4dim: \n', input_4dim)

    groups = 3
    channels = C
    affine = True
    # 次元C=9を3分割して以降の次元をまとめた上で統計量を取る. 
    # スケールパラメータγとシフトパラメータβは, C/3 = 3次元ベクトルになるが, C=9に拡張される.
    group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=affine)

    if affine:
        gamma = group_norm.weight
        beta = group_norm.bias
        print('gamma.shape', gamma.shape)
        print('gamma: \n', gamma)
        print('beta.shape', beta.shape)
        print('beta: \n', beta)


if __name__ == "__main__":
    # Batch Norm
    # test_batch_norm_1d()
    # test_batch_norm_2d()
    # test_batch_norm_3d()

    # Layer Norm
    # test_layer_norm()

    # Instance Norm
    # test_instance_norm_1d()
    # test_instance_norm_2d()
    # test_instance_norm_3d()

    # Group Norm
    test_group_norm()
