import torch
import torch.nn as nn
from typing import Dict, Union, Optional
import warnings

# 尝试导入第三方库，如果未安装则提供备用方案
try:
    from thop import profile as thop_profile

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info

    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


class ModelAnalyzer:
    """模型复杂度分析器"""

    def __init__(self, model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
        """
        初始化分析器

        Args:
            model: PyTorch 模型
            input_size: 输入尺寸 (batch_size, channels, height, width)
        """
        self.model = model
        self.input_size = input_size
        self.device = next(model.parameters()).device

    def analyze(self, verbose: bool = True) -> Dict[str, float]:
        """
        分析模型复杂度

        Returns:
            Dict 包含 'flops_g', 'params_m', 'flops', 'params'
        """
        results = {}

        # 方法1: 使用 thop (推荐，最准确)
        if THOP_AVAILABLE:
            results = self._analyze_with_thop()
            method = "thop"
        # 方法2: 使用 ptflops
        elif PTFLOPS_AVAILABLE:
            results = self._analyze_with_ptflops()
            method = "ptflops"
        # 方法3: 手动计算
        else:
            results = self._analyze_manual()
            method = "manual"

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"模型复杂度分析结果 (使用 {method} 方法)")
            print(f"{'=' * 50}")
            print(f"输入尺寸: {self.input_size}")
            print(f"总参数量 (Params): {results['params_m']:.3f} M")
            print(f"总计算量 (FLOPs): {results['flops_g']:.3f} G")
            print(f"{'=' * 50}")

        return results

    def _analyze_with_thop(self) -> Dict[str, float]:
        """使用 thop 库计算"""
        model = self.model.eval()
        dummy_input = torch.randn(self.input_size).to(self.device)

        flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)

        return {
            'flops': flops,
            'params': params,
            'flops_g': flops / 1e9,
            'params_m': params / 1e6
        }

    def _analyze_with_ptflops(self) -> Dict[str, float]:
        """使用 ptflops 库计算"""
        model = self.model.eval()

        flops, params = get_model_complexity_info(
            model,
            self.input_size[1:],  # 去掉 batch 维度
            as_strings=False,
            print_per_layer_stat=False
        )

        return {
            'flops': flops,
            'params': params,
            'flops_g': flops / 1e9,
            'params_m': params / 1e6
        }

    def _analyze_manual(self) -> Dict[str, float]:
        """
        手动计算参数量和估算 FLOPs
        注：手动 FLOPs 估算可能不如专业库准确
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 估算 FLOPs (粗略估计)
        estimated_flops = self._estimate_flops()

        return {
            'flops': estimated_flops,
            'params': total_params,
            'flops_g': estimated_flops / 1e9,
            'params_m': total_params / 1e6,
            'trainable_params_m': trainable_params / 1e6
        }

    def _estimate_flops(self) -> int:
        """
        通过遍历层来估算 FLOPs
        """
        total_flops = 0
        input_size = self.input_size

        def count_conv_flops(module, input, output):
            nonlocal total_flops
            batch_size = output.shape[0]
            output_height, output_width = output.shape[2], output.shape[3]

            # Conv FLOPs = 2 * Cin * Cout * K * K * Hout * Wout
            # 2 是因为包含乘法和加法
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups

            # 考虑分组卷积
            flops_per_position = 2 * (in_channels // groups) * out_channels * kernel_ops
            total_output_positions = batch_size * output_height * output_width

            total_flops += flops_per_position * total_output_positions

        def count_linear_flops(module, input, output):
            nonlocal total_flops
            # Linear FLOPs = 2 * in_features * out_features (乘加操作)
            total_flops += 2 * module.in_features * module.out_features * input[0].shape[0]

        def count_bn_flops(module, input, output):
            nonlocal total_flops
            # BN: 2 * num_features (减均值、除方差)
            total_flops += 2 * module.num_features * input[0].numel() // input[0].shape[1]

        # 注册钩子
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(count_conv_flops))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(count_linear_flops))
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                hooks.append(module.register_forward_hook(count_bn_flops))

        # 前向传播以触发钩子
        dummy_input = torch.randn(self.input_size).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.model(dummy_input)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return total_flops

    def print_layer_summary(self, max_layers: int = 20):
        """打印每层的信息"""
        print(f"\n{'=' * 80}")
        print(f"{'Layer Name':<40} {'Type':<20} {'Params':<15} {'Output Shape'}")
        print(f"{'-' * 80}")

        total_params = 0
        for name, module in list(self.model.named_modules())[1:max_layers + 1]:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                total_params += params
                # 尝试获取输出形状
                print(f"{name:<40} {module.__class__.__name__:<20} {params / 1e6:>10.3f} M")

        print(f"{'-' * 80}")
        print(f"显示前 {max_layers} 层，总参数量: {total_params / 1e6:.3f} M")


def analyze_pytorch_model(
        model: nn.Module,
        input_size: tuple = (1, 3, 224, 224),
        verbose: bool = True
) -> Dict[str, float]:
    """
    便捷的函数接口，用于分析 PyTorch 模型

    Args:
        model: PyTorch 模型
        input_size: 输入张量尺寸
        verbose: 是否打印详细信息

    Returns:
        包含 flops_g 和 params_m 的字典
    """
    analyzer = ModelAnalyzer(model, input_size)
    return analyzer.analyze(verbose=verbose)


def compare_models(models_dict: Dict[str, nn.Module], input_size: tuple = (1, 3, 224, 224)):
    """
    对比多个模型的复杂度

    Args:
        models_dict: {模型名: 模型实例} 的字典
        input_size: 输入尺寸
    """
    print(f"\n{'=' * 80}")
    print(f"{'Model Name':<30} {'Params (M)':<15} {'FLOPs (G)':<15} {'Efficiency'}")
    print(f"{'-' * 80}")

    results = {}
    for name, model in models_dict.items():
        analyzer = ModelAnalyzer(model, input_size)
        result = analyzer.analyze(verbose=False)
        results[name] = result

        efficiency = result['flops_g'] / result['params_m'] if result['params_m'] > 0 else 0
        print(f"{name:<30} {result['params_m']:>10.3f}    {result['flops_g']:>10.3f}    {efficiency:>8.2f}")

    print(f"{'=' * 80}")
    return results


if __name__ == "__main__":
    # 检查依赖
    print("检查依赖库...")
    if not THOP_AVAILABLE:
        print("⚠️  未安装 thop，建议: pip install thop")
    if not PTFLOPS_AVAILABLE:
        print("⚠️  未安装 ptflops，建议: pip install ptflops")

    from JASTFRNet import JASTFRNet

    model = JASTFRNet()

    # 一键分析
    result = analyze_pytorch_model(model, input_size=(1, 1, 5, 256, 256))

    # 获取数值
    flops_g = result['flops_g']  # Giga FLOPs
    params_m = result['params_m']  # Mega Params
