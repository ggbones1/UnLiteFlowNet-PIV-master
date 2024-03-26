import torch.nn.functional as F
import torch


# 定义EPE函数，计算预测流和目标流的欧氏距离
def EPE(input_flow, target_flow, mean=True):
    # Calculate the EPE along the second dimension
    EPE_map = torch.norm(target_flow - input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if mean:
        # 返回EPE_map的平均值
        return EPE_map.mean()
    else:
        # 返回EPE_map的总和除以批次大小
        return EPE_map.sum() / batch_size


# 定义realEPE函数，计算预测流和目标流的欧氏距离
def realEPE(output, target):
    # 获取目标流的批次大小、高度和宽度
    b, _, h, w = target.size()
    # 将输出流调整为与目标流相同的高度和宽度
    upsampled_output = F.interpolate(output, (h, w),
                                     mode='bilinear',
                                     align_corners=False)
    # 计算预测流和目标流的欧氏距离
    return EPE(upsampled_output, target, mean=True)
