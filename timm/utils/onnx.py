from typing import Optional, Tuple, List
import os
import torch
import onnxsim

def onnx_forward(onnx_file, example_input):
    import onnxruntime

    sess_options = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(onnx_file, sess_options)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: example_input.numpy()})
    output = output[0]
    return output


def onnx_export(
        model: torch.nn.Module,
        output_file: str,
        example_input: Optional[torch.Tensor] = None,
        training: bool = False,
        verbose: bool = False,
        check: bool = True,
        check_forward: bool = False,
        batch_size: int = 64,
        input_size: Tuple[int, int, int] = None,
        opset: Optional[int] = None,
        dynamic_size: bool = False,
        aten_fallback: bool = False,
        keep_initializers: Optional[bool] = None,
        input_names: List[str] = None,
        output_names: List[str] = None,
):
    import onnx

    if training:
        training_mode = torch.onnx.TrainingMode.TRAINING
        model.train()
    else:
        training_mode = torch.onnx.TrainingMode.EVAL
        model.eval()

    if example_input is None:
        if not input_size:
            assert hasattr(model, 'default_cfg')
            input_size = model.default_cfg.get('input_size')
        example_input = torch.randn((batch_size,) + input_size, requires_grad=training)
        print("example input shape is :", example_input.shape, training)

    # Run model once before export trace, sets padding for models with Conv2dSameExport. This means
    # that the padding for models with Conv2dSameExport (most models with tf_ prefix) is fixed for
    # the input img_size specified in this script.

    # Opset >= 11 should allow for dynamic padding, however I cannot get it to work due to
    # issues in the tracing of the dynamic padding or errors attempting to export the model after jit
    # scripting it (an approach that should work). Perhaps in a future PyTorch or ONNX versions...
    original_out = model(example_input)
    input_names = input_names or ["input0"]
    output_names = output_names or ["output0"]

    print("dynamic size is ", dynamic_size)
    if dynamic_size:
        
        dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
        # dynamic_axes['input0'][2] = 'height'
        # dynamic_axes['input0'][3] = 'width'
    else: 
        dummy_input = torch.randn(1,3,224,224)
        
        
    if aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX
    torch_out = torch.onnx.export(
        model,
        example_input if dynamic_size else dummy_input,
        output_file,
        training=training_mode,
        do_constant_folding=True, 
        export_params=True,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=keep_initializers,
        dynamic_axes=dynamic_axes if dynamic_size else None,
        opset_version=opset,
        operator_export_type=export_type
    )
    
    print("=" * 10, output_file)
    # Load the ONNX model
    
    

    # Simplify the model
    # onnx_model = onnx.load(output_file)
    # simplified_model, check = onnxsim.simplify(onnx_model)

    # # Check if the model has been simplified
    # if check:
    #     print("The model has been simplified.")
    # else:
    #     print("The model could not be simplified.")

    # # 使用 splitext() 分割文件路径和后缀
    # path_without_extension, file_extension = os.path.splitext(output_file)

    # print(f"不含后缀的文件路径: {path_without_extension}")  # 'path/to/your/file'
    # print(f"文件后缀: {file_extension}")  # '.txt'

    # simplified_model_path = path_without_extension+"_onnsim.onnx"
    # onnx.save(simplified_model, simplified_model_path)
    
    
    
    
    
    if check:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model, full_check=True)  # assuming throw on error
        if check_forward and not training:
            import numpy as np
            onnx_out = onnx_forward(output_file, example_input)
            np.testing.assert_almost_equal(torch_out.data.numpy(), onnx_out, decimal=3)
            np.testing.assert_almost_equal(original_out.data.numpy(), torch_out.data.numpy(), decimal=5)


