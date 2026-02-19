
import torch
def get_batch_class_weights(
    labels, # [Tensor]: 当前批次中的所有样本标签，形状为 [batch_size]。
    num_classes, # [int]: 数据集中总的类别数量。
    device, # [torch.device]: PyTorch设备，通常为 'cpu' 或 'cuda'。
    prev_weights=None, # [Tensor, 可选]: 上一个批次计算得到的类别权重，用于动量更新。
    smoothing_eps=1e-5, # [float]: 避免除以零的小常数。
    clamp_min=0.2, # [float]: 权重允许的最小值，用于防止权重过小。
    clamp_max=5.0, # [float]: 权重允许的最大值，用于防止权重过大。
    momentum=0.8 # [float]: 动量因子，用于平滑新计算出的权重，使其变化不那么剧烈。
):
    # 1. 计算每个类别的样本数量
    # bincount 会统计 labels 中每个元素的出现次数，minlength 确保输出张量长度为 num_classes
    # counts: [num_classes]
    counts = torch.bincount(labels, minlength=num_classes).float()

    # 2. 计算每个类别的初始权重
    # 权重与样本数量成反比，样本越少的类别，权重越高
    weights = 1.0 / (counts + smoothing_eps)

    # 3. 处理缺失类别 (counts == 0)
    # 如果某个类别在当前批次中没有出现，为其赋予一个默认权重
    # 这个权重是 1.0 / (总样本数)，以保证其权重不为无穷大，且相对较小
    weights[counts == 0] = 1.0 / (counts.sum() + smoothing_eps)

    # 4. 权重归一化和平衡
    # 将权重总和归一化到 num_classes，这使得所有类别的平均权重为 1
    weights = weights / weights.sum() * num_classes

    # 5. 应用手动调整的权重因子 (mu_k)
    # mu_k 是一个经验性的、手动设定的张量，用于进一步调整特定类别的权重
    # 在这个例子中，它似乎是为5个类别定制的，例如类别1和4可能需要更高的权重
    mu_k = torch.tensor([0.9, 1.5, 1.0, 0.7, 1.5], device=device)
    weights = weights * mu_k

    # 6. 限制权重范围
    # 将权重限制在 [clamp_min, clamp_max] 之间，防止极端值影响训练稳定性
    weights = weights.clamp(min=clamp_min, max=clamp_max)

    # 7. 应用动量平滑 (如果提供了上一个批次的权重)
    # 如果 prev_weights 存在，则使用动量公式更新权重
    # 新权重 = 上一个权重 * momentum + 新计算的权重 * (1 - momentum)
    # 这使得权重变化更加平滑，减少批次间波动的噪声
    if prev_weights is not None:
        weights = prev_weights * momentum + weights * (1 - momentum)

    # 8. 返回最终权重
    # 将最终权重张量移动到指定设备，并返回
    return weights.to(device)