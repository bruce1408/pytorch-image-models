import os
import onnx

def convert_dynamic_to_static(onnx_model_path):
    file_name, _ = os.path.splitext(onnx_model_path)
    model = onnx.load(onnx_model_path)

    # 假设我们要将 batch-size 设置为 1
    # 检查模型的第一个输入节点（通常是 batch-size）
    # 并将其维度设置为 1
    input_tensor = model.graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape
    input_shape.dim[0].dim_value = 1  # 将 batch-size 设置为 1

    # 保存修改后的模型
    onnx.save(model, file_name+"_static.onnx")


if __name__=="__main__":
    # 加载 ONNX 模型
    onnx_model_path = "/mnt/share_disk/cdd/export_onnx_models/swin_tiny_patch4_window7_224_20240201_193003.onnx"
    convert_dynamic_to_static(onnx_model_path)
    