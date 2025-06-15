import numpy as np
import os

# ========== 文件路径 ==========
file_path = r"C:\Users\tiffa\Downloads\cleanDynamic.bin"  # 你的 bin 文件路径
output_dir = r"D:\Dataset"         # 保存文件的目标文件夹

# ========== 创建输出文件夹 ==========
os.makedirs(output_dir, exist_ok=True)

# ========== 分块读取并保存 ==========
chunk_size = 10_000_000  # 每个块读取多少个 int16 数值（包括 I 和 Q）

# 计算总样本数
total_samples = os.path.getsize(file_path) // 2  # int16 总数（2字节为1个 int16）
num_chunks = int(np.ceil(total_samples / chunk_size))

with open(file_path, 'rb') as f:
    for i in range(num_chunks):
        data = np.fromfile(f, dtype=np.int16, count=chunk_size)
        if len(data) == 0:
            break  # 文件结束
        I = data[0::2]  # 偶数索引为 I
        Q = data[1::2]  # 奇数索引为 Q
        IQ = I + 1j * Q # 形成复数信号

        output_path = os.path.join(output_dir, f"chunk_{i+1}.npy")
        np.save(output_path, IQ)
        print(f"已保存：{output_path}，包含 {len(IQ)} 个复数样本")

print("全部信号分块保存完成！")

