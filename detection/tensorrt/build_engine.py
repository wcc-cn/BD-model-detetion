import tensorrt as trt


def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    # 显式批次模式
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 获取 ONNX 输入
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape  # 例如 (1,3,640,640)

    # 判断是否动态模型 (检查是否包含 -1)
    if -1 in input_shape:
        print("Dynamic ONNX detected. Using dynamic profile.")
        # 你可以自己根据需要修改 min/opt/max
        min_shape = (1, 3, 320, 320)
        opt_shape = (1, 3, 640, 640)
        max_shape = (1, 3, 1280, 1280)
    else:
        print("Static ONNX detected. Using static input profile.")
        # 静态模型必须使用相同形状
        min_shape = tuple(input_shape)
        opt_shape = tuple(input_shape)
        max_shape = tuple(input_shape)

    # 创建 profile
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 构建 engine
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print('Failed to build the engine')
        return None

    with open(engine_file_path, 'wb') as f:
        f.write(engine)

    print("Engine built successfully!")
    return engine

if __name__ == "__main__":
    # print(dir(trt))
    build_engine('../../model/best.onnx', '../../model/best.engine')
