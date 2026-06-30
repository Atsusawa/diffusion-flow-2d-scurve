# 2D S-curve 生成实验：Diffusion 与 Flow Matching

本仓库实现一个二维生成任务：从二维标准高斯分布出发，生成 S 形目标分布。实现重点是比较两类生成建模方式在同一低维任务中的训练目标、采样过程、轨迹形态和采样预算响应。

代码部分使用 `numpy` 手写 MLP、反向传播和 Adam，不依赖 PyTorch 或 Conda。幻灯片部分使用静态 HTML，可以直接在浏览器中打开。

## 目录说明

```text
scurve_diffusion_flow.py     实验主程序
requirements.txt             Python 依赖
analysis.md                  实验结果与分析记录
presentation/                正式 HTML 幻灯片
  index.html                 幻灯片入口
  shared.css                 幻灯片公共样式
  slides/                    单页 HTML
  assets/                    幻灯片图片、GIF、图表等正式素材
  vendor/katex/              本地 KaTeX 运行文件
outputs/                     实验输出，已忽略，可重新生成
screenshots/                 幻灯片截图校验产物，已忽略
```

`presentation/notes/`、`speaker*.md`、`script*.md` 等讲稿和过程笔记不作为提交内容；最终保留的是幻灯片本身及其素材。

## 运行实验

安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

快速检查：

```powershell
python .\scurve_diffusion_flow.py --quick --out outputs_quick
```

完整运行：

```powershell
python .\scurve_diffusion_flow.py
```

默认输出在 `outputs/` 中：

- `diffusion_process.gif`：Diffusion / DDIM 反向采样过程。
- `flow_matching_process.gif`：Flow Matching / ODE 采样过程。
- `comparison.png`：目标分布、Diffusion 结果、Flow Matching 结果对比。
- `metrics.json`：训练耗时、采样耗时、NFE 与 sliced Wasserstein distance。
- `samples.npz`：目标样本与两种模型生成样本。

`outputs/` 属于可再生实验产物，不纳入版本控制。幻灯片中真正使用到的图像和 GIF 已整理到 `presentation/assets/`。

## 查看幻灯片

直接打开：

```text
presentation/index.html
```

幻灯片依赖相对路径加载单页、图片和公式渲染文件。因此迁移到其他设备时，应保持整个 `presentation/` 目录结构不变，尤其是：

- `presentation/slides/`
- `presentation/assets/`
- `presentation/vendor/katex/`
- `presentation/shared.css`

当前公式渲染使用本地 KaTeX：`presentation/vendor/katex/` 已包含 `katex.min.css`、`katex.min.js` 和字体文件。只要该目录随仓库一起保留，另一台设备打开 HTML 幻灯片时不需要再安装 KaTeX，也不需要联网加载公式库。

只有在本地重新做截图校验或开发辅助工具时，才可能需要 Node 依赖；这不影响直接浏览幻灯片。

## 建模设定

目标分布是二维 S 曲线：

```text
u ~ Uniform(-2.05, 2.05)
x = 1.15 sin(1.65u) + noise
y = 0.82u + noise
```

源分布是二维标准高斯 `N(0, I)`。两种方法都从高斯样本出发，学习生成目标 S 形分布。

## Diffusion Model

训练时从真实样本 `x0` 出发构造带噪样本：

```text
xt = alpha(t) x0 + sigma(t) eps
alpha(t) = cos(pi t / 2)
sigma(t) = sin(pi t / 2)
eps ~ N(0, I)
```

网络输入 `(xt, t)`，预测注入噪声 `eps`：

```text
L = E || eps_theta(xt, t) - eps ||^2
```

采样时从高斯噪声端开始，使用确定性的 DDIM-like 更新：

```text
x0_hat = (xt - sigma(t) eps_theta(xt,t)) / alpha(t)
x_{t_next} = alpha(t_next) x0_hat + sigma(t_next) eps_theta(xt,t)
```

本实现强调“噪声预测如何参数化反向去噪过程”。由于采样器采用确定性 DDIM-like 更新，实验中的 Diffusion 采样过程不等同于带随机项的完整 DDPM 反向采样。

## Flow Matching

训练时从高斯样本 `z0` 与目标样本 `x1` 构造线性概率路径：

```text
zt = (1 - t) z0 + t x1
u_t = x1 - z0
```

网络输入 `(zt, t)`，回归路径上的速度：

```text
L = E || v_theta(zt,t) - (x1 - z0) ||^2
```

采样时求解常微分方程：

```text
dx / dt = v_theta(x,t),  x(0) ~ N(0,I)
```

默认实现使用 Euler 积分从 `t=0` 推进到 `t=1`。

## 比较指标

主要观察对象包括：

- 终态分布：生成样本是否贴近目标 S 形分布。
- 样本轨迹：同一批初始点在采样过程中如何移动。
- 局部速度场：不同空间位置的更新方向是否连贯。
- 采样预算：固定训练权重后改变 NFE，观察误差曲线。

主要数值指标：

- `sliced_wasserstein`：切片 Wasserstein 距离，越低表示生成分布越接近目标分布。
- `sample_seconds`：采样耗时。
- `nfe`：神经网络前向调用次数，可近似理解为采样计算预算。
- `train_seconds` 与训练 loss：反映训练开销和收敛情况。

本实验只能支持当前二维任务、当前网络容量和当前采样器下的比较；它不构成对两类方法的一般排序。

参考资料：<https://diffusionflow.github.io/>
