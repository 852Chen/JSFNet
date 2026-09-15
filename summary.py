# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
import numpy as np
from thop import clever_format, profile
from torchsummary import summary
# 注意：根据你的实际路径调整导入
from nets.mymodel import DeepLab


# 修复torchsummary对多输出/不规则输出的兼容问题
class ModifiedSummary:
    @staticmethod
    def summary(model, input_size, batch_size=-1, device="cuda"):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = f"{class_name}-{module_idx + 1}"
                summary[m_key] = {}

                # 处理多输出张量的情况
                if isinstance(output, (list, tuple)):
                    # 取第一个输出作为代表，或拼接形状
                    output_shape = list(output[0].size())
                else:
                    output_shape = list(output.size())

                output_shape[0] = batch_size
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                summary[m_key]["output_shape"] = output_shape

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.tensor(module.weight.size())).item()
                    if hasattr(module, "bias") and hasattr(module.bias, "size"):
                        params += torch.prod(torch.tensor(module.bias.size())).item()
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, torch.nn.Sequential)
                    and not isinstance(module, torch.nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # 检查输入尺寸
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # 创建输入张量
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        if batch_size != -1:
            for i in range(len(x)):
                x[i] = x[i].repeat(batch_size, 1, 1, 1)

        # 初始化summary字典
        summary = {}
        hooks = []

        # 注册hook
        model.apply(register_hook)

        # 前向传播
        model(*x)

        # 移除hook
        for h in hooks:
            h.remove()

        # 打印结果
        print("----------------------------------------------------------------")
        line_new = f"{'Layer (type)':<20} {'Output Shape':<30} {'Param #':<15}"
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0

        for layer in summary:
            # 跳过输出形状为不规则的层（避免numpy prod错误）
            try:
                output_shape = summary[layer]["output_shape"]
                if any(isinstance(s, (list, tuple)) for s in output_shape):
                    continue
                total_output += np.prod(output_shape)
            except:
                continue

            line_new = f"{layer:<20} {str(summary[layer]['output_shape']):<30} {summary[layer]['nb_params']:<15}"
            total_params += summary[layer]["nb_params"]
            print(line_new)

        # 统计总参数量
        total_params = sum(summary[layer]["nb_params"] for layer in summary)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print("================================================================")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {non_trainable_params:,}")
        print("----------------------------------------------------------------")
        try:
            print(f"Input size (MB): {np.prod(input_size) * batch_size * 4 / (1024 ** 2):.2f}")
            print(f"Forward/backward pass size (MB): {total_output * 4 / (1024 ** 2):.2f}")
            print(f"Params size (MB): {total_params * 4 / (1024 ** 2):.2f}")
            print(
                f"Estimated Total Size (MB): {np.prod(input_size) * batch_size * 4 / (1024 ** 2) + total_output * 4 / (1024 ** 2) + total_params * 4 / (1024 ** 2):.2f}")
        except:
            pass
        print("----------------------------------------------------------------")
        return summary


if __name__ == "__main__":
    input_shape = [512, 512]
    num_classes = 12
    backbone = 'starnet'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=8,
        pretrained=False
    ).to(device)

    # 替代原summary，使用修改后的版本
    try:
        # 尝试原torchsummary（如果模型输出简单）
        summary(model, (3, input_shape[0], input_shape[1]))
    except ValueError as e:
        print(f"原torchsummary出错，使用修改后的版本：{e}")
        ModifiedSummary.summary(model, (3, input_shape[0], input_shape[1]), device=str(device))

    # 计算FLOPs和参数量
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    # 避免模型重复to(device)
    flops, params = profile(model, (dummy_input,), verbose=False)

    # 修正FLOPs计算
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")

    # 打印结果
    print('\n==================== Model Statistics ====================')
    print(f'Total GFLOPS: {flops}')
    print(f'Total params: {params}')

    # 可选：将结果写入文件
    with open('starnet_stats.txt', 'w', encoding='utf-8') as f:
        f.write('==================== Model Statistics ====================\n')
        f.write(f'Total GFLOPS: {flops}\n')
        f.write(f'Total params: {params}\n')