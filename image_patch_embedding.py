"""画像を埋め込みベクトルに変換(画像->パッチ分割->埋め込みベクトル化: 畳み込みで一連の計算を一括で行う. 畳み込みのパラメータがEmbeddingパラメータに相当.)
Input: (N, C, H, W)
Output: (N, T, D)
P: 正方形パッチの一辺の長さ
Middle1(N, D, H/P, W/P) = Input(N, C, H, W) @ Conv2d(in_channels=C, out_channels=T, kernel_size=P, stride=P) [ = Linear as Embedding_W(P*P, D) about one patch vector(1, P*P)]
Middle2(N, D, T=`H*W/P^2`) = Middle1(N, D, H/P, W/P).flatten(dim=-2)
Output(N, T, D) = Middle2(N, D, T).transpose(1,2)
"""

from typing import *
from typing_extensions import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImagePatchEmbedding(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int, image_size: Tuple[int, int], patch_num: Tuple[int, int]):
        super(ImagePatchEmbedding, self).__init__()

        self.channels: int = in_channels  # 画像のチャンネル数
        self.embed_dim: int = embed_dim   # 埋め込みベクトルの次元数

        self.image_h: int = image_size[0] # 画像高さ
        self.image_w: int = image_size[1] # 画像幅

        self.num_patch_row: int = patch_num[0] # 縦方向のパッチ数
        self.num_patch_col: int = patch_num[1] # 横方向のパッチ数

        self.series_dim: int = self.num_patch_row * self.num_patch_col # 系列長(トークン数)

        self.patch_h: int = self.image_h // self.num_patch_row # パッチ高さ
        self.patch_w: int = self.image_w // self.num_patch_col # パッチ幅

        assert self.image_h % self.num_patch_row == 0, \
            "Not divide self.image_h with self.num_patch_row. Given is ({%d},{%d})".format(self.image_h, self.num_patch_row)
        
        assert self.image_w % self.num_patch_col == 0, \
            "Not divide self.image_w with self.num_patch_col. Given is ({%d},{%d})".format(self.image_w, self.num_patch_col)

        self.image_patch_embedding: nn.Conv2d = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            bias=True
        )


    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, D, H/P, W/P)
        patchs_as_layer2d = self.image_patch_embedding(img)
        # (N, D, H/P, W/P) -> (N, D, T=`H*W/P^2`)
        patchs_as_vector = torch.flatten(patchs_as_layer2d, start_dim=-2, end_dim=-1)
        # (N, D, T) -> (N, T, D)
        out = patchs_as_vector.transpose(1, 2)
        return out


def test_square_image_patch_embedding():
    
    # --- 動作確認用
    from matplotlib import pyplot as plt
    import numpy as np
    torch.set_printoptions(edgeitems=2, linewidth=75)
    torch.manual_seed(123)

    from torchvision import datasets
    data_path = "C:\\Users\\inoue\\Documents\\AI_Learning_Dataset\\"
    cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # fig = plt.figure(figsize=(8,3))
    # num_classes = 10
    # for i in range(num_classes):
    #     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    #     ax.set_title(class_names[i])
    #     img = next(img for img, label in cifar10 if label == i)
    #     plt.imshow(img)
    # plt.show()

    img, label = cifar10[99]
    img, label, class_names[label]
    plt.imshow(img)
    plt.show()

    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    # (1, 3, 32, 32)
    img_t = to_tensor(img).unsqueeze(0)

    SIPE = ImagePatchEmbedding(
        in_channels=3,
        embed_dim=64,
        image_size=(32, 32),
        patch_num=(4, 4)
    )

    embed_out = SIPE(img_t)

    print("SIPE:", SIPE)

    print("series_length: ", SIPE.series_dim)
    print('embed_out.shape: ', embed_out.shape)


if __name__ == "__main__":
    test_square_image_patch_embedding()