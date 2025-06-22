from pathlib import Path
import matplotlib
matplotlib.use('agg')

import torch
from matplotlib import pyplot as plt
from torch.onnx.symbolic_opset9 import tensor


def visualize_immediate_features(x, merge_mode="mean", save_dir="", index=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    f = save_dir / f"layer {index}.png"

    # Merge channels
    if merge_mode == "mean":
        merged_feature = x[0].mean(dim=0, keepdim=True).cpu()  # Channel-wise mean
    elif merge_mode == "sum":
        merged_feature = x[0].sum(dim=0).cpu()  # Channel-wise sum
    elif merge_mode == "max":
        merged_feature, _ = x[0].max(dim=0)  # Channel-wise max
        merged_feature = merged_feature.cpu()
    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}. Choose from 'mean', 'sum', 'max'.")

    # 将中间层特征图根据阈值，将特征值不高的f(x,y)置为0
    # w,h = merged_feature.shape[0], merged_feature.shape[1]
    # new_merged_feature = torch.empty(w,h)
    # for i in range(w):
    #     for j in range(h):
    #         if merged_feature[i][j] < merged_feature.mean():
    #             new_merged_feature[i][j] = 0
    #         else:
    #             new_merged_feature[i][j] = merged_feature[i][j]

    # Visualize and save
    plt.imsave(f, merged_feature, cmap='viridis')

def visualize_one_feature(x, save_dir="", index=None, Type='viridis'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    f = save_dir / f"layer {index}.png"

    x = x[0]
    # plt.figure(figsize=(6, 6))
    # plt.imshow(x, cmap="viridis")  # Use colormap for visualization
    # plt.axis("off")
    # plt.savefig(f, bbox_inches='tight')
    # plt.close()
    plt.imsave(f, x, cmap=Type)

