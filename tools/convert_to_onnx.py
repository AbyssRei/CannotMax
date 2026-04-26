import torch
import os
import __main__  # 引入当前主命名空间

# 从训练脚本中引入模型
from train import UnitAwareTransformer
# 从配置文件中引入常量，以计算特征数
from recognize import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

# 将引入的类绑定到当前的 __main__ 命名空间中
setattr(__main__, "UnitAwareTransformer", UnitAwareTransformer)

# --- 配置 ---
MODEL_PATH_PTH = "models/best_model_full.pth"
MODEL_PATH_ONNX = "models/best_model_full.onnx"
BATCH_SIZE = 1  # 作为导出时虚拟输入的 Batch Size

# 定义一个包装器，强行把模型的输出剥离成 0 维标量
class SqueezeOutputWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, left_signs, left_counts, right_signs, right_counts):
        # 拿到原始模型的输出，此时 shape 是 [1]
        out = self.model(left_signs, left_counts, right_signs, right_counts)
        # 取出第一个结果（剥去 Batch 维度），使其 shape 变成 [] (0 维标量)
        return out[0]

def main():
    print(f"正在从 {MODEL_PATH_PTH} 加载PyTorch模型...")

    if not os.path.exists(MODEL_PATH_PTH):
        print(f"错误: 未找到模型文件 '{MODEL_PATH_PTH}'。请确保模型文件存在。")
        return

    try:
        model = torch.load(MODEL_PATH_PTH, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    model.eval()

    # 把原始模型套进包装器里
    wrapper_model = SqueezeOutputWrapper(model)
    wrapper_model.eval()

    print("模型加载成功。开始构造虚拟输入...")

    num_units = MONSTER_COUNT + FIELD_FEATURE_COUNT

    # 保持 int64 类型，迎合 .exe 的要求
    dummy_left_signs = torch.zeros(BATCH_SIZE, num_units, dtype=torch.int64)
    dummy_left_counts = torch.zeros(BATCH_SIZE, num_units, dtype=torch.int64)
    dummy_right_signs = torch.zeros(BATCH_SIZE, num_units, dtype=torch.int64)
    dummy_right_counts = torch.zeros(BATCH_SIZE, num_units, dtype=torch.int64)

    dummy_input = (dummy_left_signs, dummy_left_counts, dummy_right_signs, dummy_right_counts)

    # 去掉 'output' 的动态轴设定，因为它是 0 维的
    dynamic_axes = {
        'left_signs': {0: 'batch_size'},
        'left_counts': {0: 'batch_size'},
        'right_signs': {0: 'batch_size'},
        'right_counts': {0: 'batch_size'}
    }

    input_names = ['left_signs', 'left_counts', 'right_signs', 'right_counts']
    output_names = ['output']

    print("开始导出 ONNX 模型...")

    # 导出模型
    torch.onnx.export(
        wrapper_model,               # 导出的是套了壳的 wrapper_model
        dummy_input,
        MODEL_PATH_ONNX,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    print("-" * 50)
    print(f"模型成功转换为ONNX格式，并保存为 '{MODEL_PATH_ONNX}'")
    print("-" * 50)


if __name__ == "__main__":
    main()