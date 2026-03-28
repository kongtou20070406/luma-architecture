# WSL2 Agent Environment

## 1. 当前默认环境

当前这套 Luma 实验代码默认在 **WSL2** 下开发和验证。

建议基线环境：

- Windows 11 + WSL2
- Ubuntu 22.04/24.04
- NVIDIA GPU（当前主实验机是 `RTX 5090 32GB`）
- CUDA 在 WSL2 下可用
- Python `3.12`

## 2. 推荐目录结构

建议把仓放在类似路径：

```bash
/home/<user>/ai/luma-architecture
```

并在同级或可访问位置准备额外依赖：

- `flash-linear-attention`
- 已安装的 `mamba-ssm`

注意：当前代码中有研究期路径约定，`minimind/model/model_minimind.py` 会默认在仓库同级寻找 `flash-linear-attention`。

推荐同级布局：

```bash
/home/<user>/ai/
  luma-architecture/
  flash-linear-attention/
```

## 3. 创建虚拟环境

推荐单独建一个全局实验环境：

```bash
python3 -m venv ~/.venvs/luma-global
source ~/.venvs/luma-global/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装基础依赖

先安装 PyTorch（按你自己的 CUDA 版本选择官方 wheel），然后再装仓内依赖：

```bash
cd minimind
pip install -r requirements.txt
```

## 5. 安装额外依赖

### 5.1 mamba-ssm

当前 `Mamba3` 模块依赖官方 `mamba_ssm`：

```bash
pip install mamba-ssm
```

如果 wheel 不可用，需要按官方仓说明本地编译。

### 5.2 bitsandbytes

当前优化器主线里有 `AdamW8bit`：

```bash
pip install bitsandbytes
```

### 5.3 muon-optimizer

当前训练主线使用外部 `muon` 包：

```bash
pip install muon-optimizer
```

### 5.4 flash-linear-attention

当前代码默认在仓库同级寻找 `flash-linear-attention` 源码目录。

做法：

```bash
cd /home/<user>/ai
git clone <flash-linear-attention-repo-url>
```

## 6. WSL2 注意事项

### CUDA / 驱动

- 先确认 Windows 侧 NVIDIA 驱动正常
- 再确认 WSL2 内 `nvidia-smi` 可用

### 代理 / 网络

如果你在 WSL2 里通过 Windows 侧代理上网，建议单独确认：

```bash
curl https://huggingface.co
```

如果拉取模型或数据失败，先排查 WSL2 网络与代理映射问题。

## 7. 当前不包含的内容

这个公开仓不包含：

- 私有 `luma_dataset/`
- 训练权重
- checkpoint
- 本地实验缓存

如果 agent 要复现实验，需要自行准备数据目录，并在脚本参数里显式传入路径。

## 8. 推荐先跑什么

先做最小验证：

```bash
cd minimind/scripts
python run_luma_stage12.py --help
```

然后再决定跑：

- 结构 smoke
- `128-step` 验证
- `512-step` 验证
- 正式 pretrain trainer smoke

## 9. 给后续 agent 的提醒

- 当前环境默认是 **WSL2**，不是原生 Linux 服务器。
- 某些 kernel / CUDA / Triton 问题在 WSL2 下更敏感。
- 遇到速度、共享内存或编译问题时，先确认是不是 WSL2 特有行为，再判断是不是架构问题。
