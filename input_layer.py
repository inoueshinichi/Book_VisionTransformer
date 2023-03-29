"""Encoderへの入力を整えるモジュール
Input_1 (Patch+Positional Embed Tokens): (N, T, D)
Input_2 (Class Token): (N, 1, D)
Output: (N, T+1=`T`, D)

Images ---> patch embedding tokens --------> cat -----> +----> (Encoder) 
                                              |         |
            class token ----------------------|         |
                                                        |
            positional embedding tokens -----------------
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_patch_embedding import ImagePatchEmbedding


class ViTInputLayer(nn.Module):
    
    def __init__(self, in_channels: int, embed_dim: int, 
                 image_size: Tuple[int, int], patch_num: Tuple[int, int]):
        super(ViTInputLayer, self).__init__()

        self.image_patch_embedding = ImagePatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_num=patch_num
        )

        self.series_dim: int = self.image_patch_embedding.series_dim
        self.embed_dim: int = embed_dim
        self.batch_size: Optional[int] = None

        self.params: nn.ParameterDict = nn.ParameterDict()

        # Class Token Embedding (データサンプル毎に異なる. 後でバッチ数分に拡張)
        self.params['class_token'] = nn.Parameter(
            torch.randn((1, 1, self.embed_dim), dtype=torch.float32) # 学習パラメータ
        )

        # Positional Embedding (バッチ数を通して共通のパラメータ)
        self.params['positional_embedding'] = nn.Parameter(
            torch.randn((1, self.series_dim + 1, self.embed_dim), dtype=torch.float32) # 学習パラメータ
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, C, H, W)

        # バッチ数
        self.batch_size = x.shape[0]

        # (N, T, D)
        patch_tokens = self.image_patch_embedding(x)

        """バッチ数分にパラメータ数を増やす.
        Class Token Embedding
        """
        # (1, 1, D) -> (N, 1, D)
        self.params['class_token'] = self.params['class_token'].repeat(repeats=(self.batch_size, 1, 1))

        # 系列長に調整
        # concatenate[(N, 1, D), (N, T, D)] -> (N, 1+T=`T`, D)
        vit_tokens = torch.cat([self.params['class_token'], patch_tokens], dim=1)

        # 位置埋め込み
        vit_tokens = vit_tokens + self.params['positional_embedding']

        return vit_tokens
    

def test_vit_input_layer():
    # 動作確認

    from matplotlib import pyplot as plt
    import numpy as np
    torch.set_printoptions(edgeitems=2, linewidth=75)
    torch.manual_seed(123)

    from torchvision import datasets
    data_path = "C:\\Users\\inoue\\Documents\\AI_Learning_Dataset\\"
    cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    img, label = cifar10[99]
    img, label, class_names[label]
    plt.imshow(img)
    plt.show()

    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    # (1, 3, 32, 32)
    img_t = to_tensor(img).unsqueeze(0)

    vit_input_layer = ViTInputLayer(
        in_channels=3,
        embed_dim=64,
        image_size=(32, 32),
        patch_num=(4, 4)
    )

    print("vit_input_layer: ", vit_input_layer)

    p_numel_list = [ p.numel() for p in vit_input_layer.parameters()]
    print(f"sum(p_numel_list): {sum(p_numel_list)}, p_numel_list: {p_numel_list}")

    output = vit_input_layer(img_t)
    print('output.shape: ', output.shape)


if __name__ == "__main__":
    test_vit_input_layer()