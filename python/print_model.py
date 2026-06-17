import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from safetensors.torch import safe_open

file_path = '../data/qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors'

try:
    with safe_open(file_path, framework="pt") as f:
        metadata = f.metadata()
        print(f"--- 模型元数据 ---")
        print(metadata if metadata else "无元数据")

        print(f"\n--- 张量结构 (Total: {len(f.keys())} layers) ---")
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"层名: {key:<50} | 形状: {tensor.shape} | 类型: {tensor.dtype}")

except Exception as e:
    print(f"读取失败: {e}")