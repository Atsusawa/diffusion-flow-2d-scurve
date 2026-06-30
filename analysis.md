# Diffusion Model 与 Flow Matching 实验分析

## 实验设置

- 任务：从二维标准高斯 `N(0, I)` 生成二维 S 形分布。
- 代码：`scurve_diffusion_flow.py`，纯 `numpy + Pillow` 实现，不依赖 PyTorch/Conda。
- 模型：两种方法都使用同一个小型 MLP，输入为二维位置 `x` 和时间嵌入 `t`，输出二维向量。
- 采样预算：默认两者都是 `80` 次网络函数调用，便于公平比较。

## 运行结果

本机默认配置运行结果位于 `outputs/`：

| 方法 | 训练耗时 | 采样耗时 | NFE | Sliced Wasserstein |
| --- | ---: | ---: | ---: | ---: |
| Diffusion/DDIM | 5.44 s | 0.54 s | 80 | 0.2472 |
| Flow matching | 5.17 s | 0.61 s | 80 | 0.0601 |

`Sliced Wasserstein` 越低，表示生成样本分布越接近目标分布。当前实验中，flow matching 明显更贴近 S 曲线；diffusion 能学到整体形状，但生成点更散，离群点更多。

## Diffusion 原理

Diffusion 先定义从真实数据到高斯噪声的前向加噪过程：

```text
xt = alpha(t) x0 + sigma(t) eps
alpha(t) = cos(pi t / 2)
sigma(t) = sin(pi t / 2)
eps ~ N(0, I)
```

训练时让网络根据 `(xt, t)` 预测噪声 `eps`：

```text
L_diffusion = E || eps_theta(xt, t) - eps ||^2
```

采样时从高斯噪声出发，使用 DDIM 确定性更新逐步去噪：

```text
x0_hat = (xt - sigma(t) eps_theta(xt,t)) / alpha(t)
x_{t_next} = alpha(t_next) x0_hat + sigma(t_next) eps_theta(xt,t)
```

直观理解：模型学习“当前 noisy 点里有多少噪声”，采样时反复估计并移除噪声。

## Flow Matching 原理

Flow matching 直接构造源分布样本 `x0` 与目标分布样本 `x1` 之间的连续路径：

```text
xt = (1 - t) x0 + t x1
u_t = x1 - x0
```

训练时让网络预测路径上的速度场：

```text
L_flow = E || v_theta(xt, t) - (x1 - x0) ||^2
```

采样时从高斯点出发，沿速度场积分：

```text
dx / dt = v_theta(x, t)
```

直观理解：模型学习“粒子此刻应该往哪里移动”，采样就是把高斯粒子连续搬运到 S 曲线。

## 差异分析

Diffusion 的优势是训练目标稳定、理论和工程生态成熟。它将复杂生成问题拆成许多去噪子问题，适合高维图像等任务。但采样常常需要多步迭代；如果网络容量、训练步数或采样步数不足，容易留下较宽的分布带和离群点。

Flow matching 在这个二维任务上更直接：训练目标就是速度场，采样过程就是 ODE 积分，因此更容易学出清晰的 S 曲线。它的局限是路径设计很重要；当前实现使用随机独立配对的线性路径，不是最优传输路径，在更复杂数据上可能需要更好的 coupling、路径或 ODE solver。

在本实验里，两者训练耗时接近，采样 NFE 相同。Flow matching 的分布指标更低，说明在这个低维映射任务上更高效、更贴合任务结构。Diffusion 的动态 GIF 更像逐步“从噪声中显影”，Flow matching 的 GIF 更像粒子沿连续速度场被推到目标曲线上。

## 可视化文件

- `outputs/diffusion_process.gif`：Diffusion/DDIM 反向去噪过程。
- `outputs/flow_matching_process.gif`：Flow matching ODE 粒子流过程。
- `outputs/comparison.png`：目标分布、Diffusion 生成、Flow matching 生成三栏图。
- `outputs/metrics.json`：可复现实验指标。
