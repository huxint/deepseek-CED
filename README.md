# Tiny CED

用 **Python 3.14 + PyTorch** 从零训练的小型 Causal Encoder-Decoder 语言模型，复现 DeepSeek-V4.1-Flash 的 **CED 数据流、局部 SWA、encoder 生成 decoder 全局 KV，以及 decoder bounded replay**。

全部参数随机初始化。架构对应关系和简化范围见 [论文与实现笔记](docs/architecture.md)。

| 配置 | 参数量 | Encoder + Decoder | 隐藏维度 | SWA 窗口 | 上下文上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [micro](configs/micro.json) | 172,800 | 2 + 2 | 64 | 32 | 256 |
| [tiny](configs/tiny.json) | 1,189,888 | 3 + 3 | 128 | 64 | 512 |

两者均使用 RMSNorm、RoPE、GQA、SwiGLU 和输入/输出 embedding 权重绑定。

## 安装

需要 Python 3.14 和 [uv](https://docs.astral.sh/uv/)。PyTorch wheel 自带所需 CUDA 运行库，无需安装 CUDA Toolkit。

```bash
uv venv --python 3.14
uv pip install --python .venv/bin/python \
  'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
python scripts/check_env.py
```

只用 CPU 时把 `cu128` 换成 `cpu`。

## 训练

```bash
python -m ced.inspect --config configs/micro.json
python -m ced.prepare --demo --out-dir data/demo
python -m ced.train --data data/demo --out runs/micro --steps 500
```

`--device auto` 优先选 CUDA，`--precision auto` 在支持的设备上用 BF16。指标写入 `runs/micro/metrics.jsonl`。

Tokenizer 使用 **256 个 UTF-8 字节 + BOS/EOS**，无需下载词表。中文通常一个汉字占三个 token。自带的示例语料是原创模板，只用于验证流程。

用自己的语料，支持 UTF-8 文本（按空行分段）或每行含 `{"text": "..."}` 的 JSONL：

```bash
python -m ced.prepare --input corpus.txt --out-dir data/local
python -m ced.train --data data/local --out runs/local
```

## 续训

```bash
python -m ced.train --data data/demo --out runs/micro \
  --resume runs/micro/last.pt --steps 1000
```

每次验证保存 `last.pt`，验证 loss 改善时保存 `best.pt`。`--steps` 是包含已完成步骤的总步数；续训需沿用原来的模型配置和优化参数。

## 生成

```bash
python -m ced.generate --checkpoint runs/micro/best.pt \
  --prompt '小猫坐在' --max-new-tokens 96
```

`--temperature 0` 为贪心解码。`--prefill bounded` 是近似预填充，结果与完整计算可能不同。prompt 长度加生成长度须在配置的上下文上限内。

## 代码入口

| 文件 | 内容 |
| --- | --- |
| [ced/model.py](ced/model.py) | 注意力、Encoder/Decoder、全局 KV、exact/bounded prefill、增量 decode |
| [ced/config.py](ced/config.py) | 不可变模型配置 |
| [ced/train.py](ced/train.py) | 训练配置、AdamW、梯度累积、AMP、验证、续训 |
| [ced/prepare.py](ced/prepare.py) / [ced/data.py](ced/data.py) | 语料准备、内存映射与采样 |
| [ced/generate.py](ced/generate.py) | 加载 checkpoint 和文本生成 |
| [tests](tests) | 因果性、缓存一致性、训练与续训验证 |

```bash
python -m unittest discover -s tests -v
```

开发工具：

```bash
uv pip install --python .venv/bin/python ruff pyright
ruff check ced scripts tests && ruff format --check ced scripts tests
pyright --pythonpath .venv/bin/python
```

## 许可证

[MIT](LICENSE)
