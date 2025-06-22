# Q1:RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip

# 这个错误表明 Jittor 正在尝试从 Tsinghua University 的云存储下载 CUTLASS，但该链接已经失效（返回 404）。我们需要手动下载并安装 CUTLASS。

### **解决方案（手动安装 CUTLASS）**
#### **1. 清除旧的缓存**
```bash
rm -rf /root/.cache/jittor/cutlass
mkdir -p /root/.cache/jittor/cutlass
cd /root/.cache/jittor/cutlass
```

#### **2. 从 NVIDIA 官方 GitHub 下载 CUTLASS**
```bash
wget https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.5.0.tar.gz -O cutlass.tar.gz
tar -xzf cutlass.tar.gz
mv cutlass-2.5.0 cutlass
```

#### **3. 修改 Jittor 源码，跳过自动下载**
由于 Jittor 仍然会尝试从 Tsinghua 镜像下载，我们可以修改其源码，使其直接使用我们手动安装的 CUTLASS。

找到 Jittor 的 `compile_extern.py` 文件（通常在 `/root/miniconda3/lib/python3.X/site-packages/jittor/compile_extern.py`），修改 `install_cutlass` 函数：
```python
def install_cutlass(root_folder):
    cutlass_path = os.path.join(root_folder, "cutlass")
    if os.path.isdir(cutlass_path):
        return  # 如果已经存在，直接返回
    os.makedirs(root_folder, exist_ok=True)
    print("[Jittor] Using manually installed CUTLASS at", cutlass_path)
    return
```

#### **4. 运行训练脚本**
```bash
python /root/autodl-tmp/project/train.py
```

### **替代方案（禁用 CUTLASS）**
如果不想手动安装，可以直接禁用 CUTLASS（性能可能略有下降）：
```bash
JT_NO_CUTLASS=1 python /root/autodl-tmp/project/train.py
```

### **原因分析**
- **原问题**：Jittor 默认从 Tsinghua 镜像下载 CUTLASS，但该链接已失效（404）。
- **解决方案**：
  - **手动下载**：直接从 NVIDIA GitHub 获取 CUTLASS。
  - **修改源码**：避免 Jittor 尝试自动下载。
  - **禁用 CUTLASS**（备选方案）。

这样应该可以解决问题，让训练正常进行。 🚀


# Q2:version `GLIBCXX_3.4.30‘ not found
https://blog.csdn.net/CCCDeric/article/details/129292944

# Q3:_pickle.UnpicklingError: STACK_GLOBAL requires str
原因：Jittor版本的yolov5和PyTorch版本的yolov5，对于同一个数据集生成的cache不同。换另一个框架训练时需要先删除原有的cache

# Q4:离线训练脚本
```bash
# YOLOv5 Jittor
nohup python -u /root/autodl-tmp/project/YOLOv5Jittor/train.py > /root/autodl-tmp/project/YOLOv5Jittor/output.log 2>&1 &
tail -f /root/autodl-tmp/project/YOLOv5Jittor/output.log
```
```bash
# YOLOv5 PyTorch
nohup python -u /root/autodl-tmp/project/YOLOv5Pytorch/train.py > /root/autodl-tmp/project/YOLOv5Pytorch/output.log 2>&1 &
tail -f /root/autodl-tmp/project/YOLOv5Pytorch/output.log
```