# Awesome AI4PDE [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Awesome AI4PDE is a curated list of resources and literature focusing on the intersection of Artificial Intelligence and Partial Differential Equations (PDEs). 

More information about the AI4PDE can be found at [AI4PDE](https://ai4pde.notion.site/).

- [1. Solving](#1-solving)
- [2. Inverse](#2-inverse)
- [3. Discovery](#3-discovery)
- [4. Analysis](#4-analysis)
- [5. Data \& Benchmarks](#5-data--benchmarks)
- [6. Applications](#6-applications)

## 1. Solving

**1. ANaGRAM: A Natural Gradient Relative to Adapted Model for efficient PINNs learning**: \[[ICLR2025](https://openreview.net/forum?id=o1IiiNIoaA)\] \[[CODE]()\] 

Tags: Functional Analysis, Green’s Function, Optimization, PINN

"ANaGRAM enhances PINN training efficiency by introducing a reduced-complexity natural gradient optimization method grounded in functional analysis and Green’s functions."

"ANaGRAM 通过结合泛函分析与格林函数理论，引入低复杂度的自然梯度优化方法，有效提升 PINN 的训练效率与稳定性。"

**2. CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems**: \[[ICLR2025](https://openreview.net/forum?id=PiHGrTTnvb)\] \[[CODE]()\] 

Tags: Diffusion

"CL-DiffPhyCon introduces an asynchronous diffusion-based closed-loop control method for PDE systems, achieving efficient and adaptive real-time control."

"CL-DiffPhyCon 提出了一种基于异步扩散的闭环控制方法，实现高效且自适应的 PDE 系统控制。"

**3. ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks**: \[[ICLR2025](https://openreview.net/forum?id=APojAzJQiq)\] \[[CODE]()\] 

Tags: Optimization, PINN

"ConFIG introduces a conflict-free gradient optimization method for PINNs, ensuring balanced and efficient training across multiple loss terms."

"ConFIG 提出了一种无冲突梯度优化方法，有效平衡 PINN 训练中的损失项，提升优化效率和准确性。"

**4. CViT: Continuous Vision Transformer for Operator Learning**: \[[ICLR2025](https://openreview.net/forum?id=cRnCcuLvyr)\] \[[CODE]()\] 

Tags: Multi-Scale, Neural Operator, Transformer

"CViT bridges operator learning and vision transformers, enabling efficient multi-scale PDE modeling with state-of-the-art accuracy."

"CViT 将算子学习与视觉 Transformer 结合，实现高效多尺度 PDE 建模，并在多个基准测试中达到最先进的精度。"

**5. Deep Learning Alternatives Of The Kolmogorov Superposition Theorem**: \[[ICLR2025](https://openreview.net/forum?id=SyVPiehSbg)\] \[[CODE]()\] 

Tags: PINN

"ActNet refines KST-based neural architectures, improving their efficiency in PINNs and PDE solving."

"ActNet 优化了基于 KST 的神经网络架构，提高了其在 PINN 和 PDE 求解中的效率。"

**6. Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics**: \[[ICLR2025](https://openreview.net/forum?id=VsxbWTDHjh)\] \[[CODE]()\] 

Tags: 3D, Fluid, Neural Operator

"Fengbo introduces a Clifford Algebra-based neural operator pipeline for solving 3D PDEs in CFD, achieving competitive accuracy with high efficiency and interpretability."

"本文提出基于 Clifford 代数 的 Fengbo 神经算子管道，用于求解 3D CFD PDEs，在高效性和可解释性方面表现优越，同时具备竞争性精度。"

**7. Generalizable Motion Planning via Operator Learning**: \[[ICLR2025](https://openreview.net/forum?id=UYcUpiULmT)\] \[[CODE]()\] 

Tags: Neural Operator, Super-Resolution

"This paper introduces PNO, a neural operator approach for solving the Eikonal PDE in motion planning, achieving resolution-invariant and generalizable path planning."

"本文提出 PNO，一种基于神经算子的运动规划方法，通过求解 Eikonal PDE，实现分辨率不变性和可泛化路径规划。"

**8. Generating Physical Dynamics under Priors**: \[[ICLR2025](https://openreview.net/forum?id=eNjXcP6C0H)\] \[[CODE]()\] 

Tags: Diffusion, PINN, Physical Priors

"This paper presents a diffusion-based generative model incorporating physical priors to generate physically feasible dynamics, enhancing realism and accuracy."

"本文提出了一种融合物理先验的扩散生成模型，以生成符合物理规律的动态，提高了仿真精度和真实感。"

**9. Gradient-Free Generation for Hard-Constrained Systems**: \[[ICLR2025](https://openreview.net/forum?id=teE4pl9ftK)\] \[[CODE]()\] 

Tags: Gradient-Free, Hard Constraints, Zero-Shot

"This paper introduces a gradient-free, zero-shot generative sampling framework that enforces hard constraints in PDE systems while preserving distribution accuracy."

"本文提出了一种梯度无关的零样本生成框架，确保PDE系统中的硬约束严格满足，同时保持分布的准确性。"

**10. GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling**: \[[ICLR2025](https://openreview.net/forum?id=Fur0DtynPX)\] \[[CODE]()\] 

Tags: Neural Fields

"MARBLE enhances neural field-based PDE modeling by combining GridMix spatial modulation with domain augmentation, improving both global structure learning and local detail preservation."

"MARBLE 结合 GridMix 空间调制与域增强技术，提升神经场在 PDE 建模中的全局结构学习与局部细节捕捉能力。"

**11. KAN: Kolmogorov–Arnold Networks**: \[[ICLR2025](https://openreview.net/forum?id=Ozo7qJ5vZi)\] \[[CODE]()\] 

Tags: Interpretable, Kolmogorov-Arnold

"KANs enhance interpretability and efficiency in function approximation and PDE solving by introducing learnable activation functions on edges, but require further scaling improvements."

"KANs 通过在边上引入可学习的激活函数，提高了函数逼近和 PDE 求解的可解释性和效率，但仍需优化以提升大规模训练能力。"

**12. Learning a Neural Solver for Parametric PDE to Enhance Physics-Informed Methods**: \[[ICLR2025](https://openreview.net/forum?id=jqVj8vCQsT)\] \[[CODE]()\] 

Tags: PINN

"This work introduces a learned solver to stabilize and accelerate the optimization of physics-informed methods for parametric PDEs, significantly improving convergence and generalization."

"本文提出了一种学习型求解器，通过优化物理损失梯度来加速和稳定参数化 PDE 的求解，提高了收敛性和泛化能力。"

**13. Learning Spatiotemporal Dynamical Systems from Point Process Observations**: \[[ICLR2025](https://openreview.net/forum?id=37EXtKCOkn)\] \[[CODE]()\] 

Tags: Temporal, Variational Inference

"This work introduces a novel framework for learning spatiotemporal dynamics from randomly observed data, integrating neural ODEs and neural point processes for robust and efficient modeling."

"本文提出了一种从随机采样数据学习时空动力学的新框架，结合神经 ODE 和神经点过程，实现了更强的泛化性和计算效率。"

**14. Learning to Solve Differential Equation Constrained Optimization Problems**: \[[ICLR2025](https://openreview.net/forum?id=VeMC6Bn0ZB)\] \[[CODE]()\] 

Tags: Optimization

"This work introduces a learning-based framework for efficiently solving DE-constrained optimization problems, integrating neural differential equations with proxy optimization for real-time control and enhanced precision."

"本文提出了一种基于学习的方法来高效求解微分方程约束优化问题，结合神经微分方程和优化代理，实现了实时控制和高精度优化。"

**15. Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups**: \[[ICLR2025](https://openreview.net/forum?id=7PLpiVdnUC)\] \[[CODE]()\] 

Tags: Lie Algebra

"This work introduces Lie Algebra Canonicalization (LieLAC), an equivariant neural operator framework that enables PDE solvers to leverage non-compact Lie group symmetries without requiring full group structure knowledge."

"本文提出了李代数规范化 (LieLAC) 方法，使 PDE 求解器能够利用非紧李群对称性，无需完整群结构知识，即可增强等变性和泛化能力。"

**16. Metamizer: A Versatile Neural Optimizer for Fast and Accurate Physics Simulations**: \[[ICLR2025](https://openreview.net/forum?id=60TXv9Xif5)\] \[[CODE]()\] 

Tags: Neural Optimizer

"Metamizer introduces a neural optimizer that accelerates PDE solving by learning an adaptive descent strategy, demonstrating strong generalization to unseen equations."

"Metamizer 提出了一种神经优化器，通过学习自适应下降策略加速 PDE 求解，并展现出对未见方程的强泛化能力。"

**17. Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout**: \[[ICLR2025](https://openreview.net/forum?id=3ep9ZYMZS3)\] \[[CODE]()\] 

Tags: Hybrid, RL

"HyPER introduces a model-agnostic reinforcement learning framework that intelligently invokes a physics simulator for error correction, significantly improving surrogate PDE model rollouts."

"HyPER 提出了一种模型无关的强化学习框架，可智能调用物理模拟器进行误差修正，大幅提升神经替代模型在 PDE 预测中的稳定性和泛化能力。"

**18. On the Benefits of Memory for Modeling Time-Dependent PDEs**: \[[ICLR2025](https://openreview.net/forum?id=o9kqa5K3tB)\] \[[CODE]()\] 

Tags: Benchmark, High-Frequency, Neural Operator, Temporal

"MemNO effectively integrates memory into neural operators, significantly improving time-dependent PDE modeling, particularly under low-resolution or noisy observations."

"MemNO 在神经算子中引入记忆机制，在低分辨率或噪声观测条件下显著提升时间依赖 PDE 预测性能。"

**19. PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems**: \[[ICLR2025](https://openreview.net/forum?id=fU8H4lzkIm)\] \[[CODE]()\] 

Tags: Boundary, GNN, Temporal

"PhyMPGN embeds physics-aware message passing and Laplacian operators into a graph network, enabling accurate spatiotemporal PDE modeling on irregular meshes."

"PhyMPGN 结合物理感知消息传递与拉普拉斯算子，在不规则网格上实现高精度时空 PDE 建模。"

**20. Physics-aligned field reconstruction with diffusion bridge**: \[[ICLR2025](https://openreview.net/forum?id=D042vFwJAM)\] \[[CODE]()\] 

Tags: Boundary

"PalSB employs a physics-aligned diffusion bridge to reconstruct physical fields from sparse measurements, achieving higher accuracy and compliance with physical constraints."

"PalSB 采用物理对齐的扩散桥机制，从稀疏观测数据重建物理场，提升精度并确保物理一致性。"

**21. Physics-Informed Diffusion Models**: \[[ICLR2025](https://openreview.net/forum?id=tpYeermigp)\] \[[CODE]()\] 

Tags: Diffusion, PINN

"This work unifies denoising diffusion models with physics-informed learning, ensuring generated data adheres to PDE constraints while reducing residual errors and mitigating overfitting."

"本文将扩散模型与物理信息学习相结合，使生成数据严格遵循 PDE 约束，同时降低残差误差并缓解过拟合问题。"

**22. Physics-Informed Neural Predictor**: \[[ICLR2025](https://openreview.net/forum?id=vAuodZOQEZ)\] \[[CODE]()\] 

Tags: Fluid, Multi-Physics, PINN

"This work integrates physics equations into a neural predictor, achieving superior long-term forecasting and generalization for fluid dynamics."

"本文将物理方程嵌入神经预测器，实现了流体动力学的高精度长期预测，并具备出色的时空泛化能力。"

**23. PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations**: \[[ICLR2025](https://openreview.net/forum?id=y5B0ca4mjt)\] \[[CODE]()\] 

Tags: High-Frequency, PINN

"PIGs introduce learnable Gaussian feature embeddings to dynamically adjust parametric mesh representations, significantly improving PDE-solving efficiency and accuracy over PINNs."

"PIGs 通过可学习的高斯特征嵌入动态调整参数网格，在提高 PDE 求解精度的同时，实现更高效的计算。"

**24. PIORF: Physics-Informed Ollivier-Ricci Flow for Long–Range Interactions in Mesh Graph Neural Networks**: \[[ICLR2025](https://openreview.net/forum?id=qkBBHixPow)\] \[[CODE]()\] 

Tags: GNN, Mesh

"PIORF introduces a novel physics-informed graph rewiring method based on Ollivier–Ricci curvature, improving long-range interactions and mitigating over-squashing in fluid simulations."

"PIORF 提出了一种基于 Ollivier–Ricci 曲率的物理增强图重连方法，提高流体模拟中长程相互作用的建模能力，并缓解 GNN 的信息压缩问题。"

**25. Progressively Refined Differentiable Physics**: \[[ICLR2025](https://openreview.net/forum?id=9Fh0z1JmPU)\] \[[CODE]()\] 

Tags: Differential Operator

"PRDP enables efficient neural network training by progressively refining differentiable physics solvers, reducing computational costs without compromising accuracy."

"PRDP 通过逐步细化可微物理求解器，提高神经网络训练效率，在降低计算成本的同时保持准确性。"

**26. Score-based free-form architectures for high-dimensional Fokker-Planck equations**: \[[ICLR2025](https://openreview.net/forum?id=5qg6JPSgCj)\] \[[CODE]()\] 

Tags: High-dimensional, PINN

"This work introduces FPNN, a novel deep learning framework for high-dimensional Fokker-Planck equations, using score PDE loss to separate density learning and normalization, achieving significant efficiency and accuracy improvements."

"本文提出 FPNN，一种求解高维 Fokker-Planck 方程的深度学习框架，利用 Score PDE Loss 进行密度学习与归一化的解耦，实现了显著的计算效率和精度提升。"

**27. Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations**: \[[ICLR2025](https://openreview.net/forum?id=DPzQ5n3mNm)\] \[[CODE]()\] 

Tags: Inverse, Neural Operator

"This work introduces SC-FNO, a sensitivity-aware enhancement of Fourier Neural Operators that improves accuracy in forward PDE solving and inverse problems, ensuring robustness under sparse data and concept drift."

"本文提出了 SC-FNO，一种具有敏感度约束的 Fourier 神经算子，提升了 PDE 求解和逆问题的精度，并在稀疏数据和概念漂移情况下保持稳定性。"

**28. SINGER: Stochastic Network Graph Evolving Operator for High Dimensional PDEs**: \[[ICLR2025](https://openreview.net/forum?id=wVADj7yKee)\] \[[CODE]()\] 

Tags: GNN, High-dimensional, Neural Operator

"This work introduces SINGER, a stochastic graph-based framework for solving high-dimensional PDEs, ensuring stability, generalization, and theoretical guarantees."

"本文提出 SINGER，一种基于随机图神经网络的高维 PDE 求解框架，具有稳定性、泛化性和理论保证。"

**29. Solving Differential Equations with Constrained Learning**: \[[ICLR2025](https://openreview.net/forum?id=5KqveQdXiZ)\] \[[CODE]()\] 

Tags: PINN

"SCL reformulates PDE solving as a constrained learning problem, integrating prior knowledge while reducing reliance on hyperparameter tuning for improved accuracy and efficiency."

"SCL 将 PDE 求解重新表述为约束学习问题，融合先验知识并减少超参数调优需求，从而提升精度和计算效率。"

**30. Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows**: \[[ICLR2025](https://openreview.net/forum?id=MKP1g8wU0P)\] \[[CODE]()\] 

Tags: Fluid, Neural Operator, Super-Resolution

"This work introduces Spectral-Refiner, a spatiotemporal Fourier neural operator with spectral fine-tuning, significantly improving the accuracy and efficiency of turbulence modeling."

"本文提出 Spectral-Refiner，一种结合时空 Fourier 神经算子和谱精炼的 PDE 求解方法，大幅提升湍流建模的精度和计算效率。"

**31. Text2PDE: Latent Diffusion Models for Accessible Physics Simulation**: \[[ICLR2025](https://openreview.net/forum?id=Nb3a8aUGfj)\] \[[CODE]()\] 

Tags: Diffusion, Mesh, Text-to-PDE

"This work introduces Text2PDE, a latent diffusion-based framework for physics simulation, enabling efficient and interpretable PDE solving with text or physics conditioning."

"本文提出 Text2PDE，一种基于潜在扩散模型的物理模拟框架，通过文本或物理条件高效求解 PDE，并提升可解释性和泛化能力。"

**32. Truncation Is All You Need: Improved Sampling Of Diffusion Models For Physics-Based Simulations**: \[[ICLR2025](https://openreview.net/forum?id=0FbzC7B9xI)\] \[[CODE]()\] 

Tags: Diffusion, Efficiency, Fluid, Precision

"This work accelerates diffusion model-based physics simulations by introducing Truncated Sampling Models and Iterative Refinement, achieving high-fidelity predictions with reduced computation."

"本文提出截断采样模型 (TSM) 和迭代细化 (IR) 方法，显著加速扩散模型在物理仿真中的采样过程，同时保持高精度预测。"

**33. Wavelet Diffusion Neural Operator**: \[[ICLR2025](https://openreview.net/forum?id=FQhDIGuaJ4)\] \[[CODE]()\] 

Tags: Diffusion, Multi-Resolution, Wavelets

"WDNO leverages wavelet-domain diffusion and multi-resolution training to achieve superior PDE simulation and control, excelling in handling abrupt changes and high-resolution generalization."

"WDNO 结合小波域扩散与多分辨率训练，在 PDE 仿真与控制任务中表现卓越，尤其擅长处理突变态与高分辨率泛化问题。"

**34. Laplace neural operator for solving differential equations**: \[[Nature Machine Intelligence2024](https://www.nature.com/articles/s42256-024-00844-4)\] \[[CODE](https://github.com/qianyingcao/Laplace-Neural-Operator)\] 

Tags: Laplace, Neural Operator

"Laplace Neural Operator (LNO) enhances PDE solving by leveraging Laplace-domain transformations, improving extrapolation, interpretability, and scalability for large-scale simulations."

"Laplace 神经算子 (LNO) 通过 拉普拉斯域变换 提升 PDE 求解能力，在外推、可解释性和大规模仿真方面表现优异。"

**35. Neural Manifold Operators for Learning the Evolution of Physical Dynamics**: \[[KDD2024](https://dl.acm.org/doi/10.1145/3637528.3671779)\] \[[CODE](https://github.com/AI4EarthLab/Neural-Manifold-Operators)\] 

Tags: Manifold, Neural Operator

"Neural Manifold Operator (NMO) introduces an adaptive dimensionality reduction technique for operator learning, enabling efficient and physically consistent modeling of high-dimensional physical dynamics."

"Neural Manifold Operator (NMO) 提出了一种自适应降维算子学习方法，实现 高效、物理一致 的 高维物理动力学建模。"

**36. Koopman neural operator as a mesh-free solver of non-linear partial differential equations**: \[[Journal of Computational Physics2024](https://www.sciencedirect.com/science/article/abs/pii/S0021999124004431)\] \[[CODE](https://github.com/Koopman-Laboratory/KoopmanLab)\] 

Tags: Neural Operator

""

""

**37. Koopman operator learning using invertible neural networks**: \[[Journal of Computational Physics2024](https://www.sciencedirect.com/science/article/pii/S0021999124000445)\] \[[CODE]()\] 

Tags: Neural Operator

""

""

**38. BENO: Boundary-embedded Neural Operators for Elliptic PDEs**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18389)\] \[[CODE](https://github.com/AI4Science-WestlakeU/beno.git)\] 

Tags: Boundary, GNN, Neural Operator, Transformer

"BENO effectively embeds boundary information into neural operator architecture, providing a significant leap in accuracy for elliptic PDEs with complex boundaries."

"BENO 有效地将边界信息嵌入神经算子结构，在处理具有复杂边界的椭圆型 PDE 时显著提升了准确性。"

**39. Better Neural PDE Solvers Through Data-Free Mesh Movers**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18088)\] \[[CODE](https://github.com/Peiyannn/MM-PDE.git)\] 

Tags: Mesh, Unsupervised

"By leveraging a data-free mesh adapter trained on the Monge-Ampère equation, this method eliminates costly mesh-label requirements and significantly enhances PDE-solving accuracy in dynamic systems."

"通过使用基于 Monge-Ampère 方程的数据无关网格自适应器 (data-free mesh adapter)，本方法无需昂贵的网格标注数据就能显著提升动态系统中的 PDE 解算精度。"

**40. Learning semilinear neural operators: A unified recursive framework for prediction and data assimilation**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18401)\] \[[CODE]()\] 

Tags: Neural Operator

"This unified neural operator framework effectively handles semilinear PDE evolution over long horizons and facilitates data assimilation from noisy, sparse measurements."

""

**41. MgNO: Efficient Parameterization of Linear Operators via Multigrid**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/19328)\] \[[CODE]()\] 

Tags: Multigrid, Neural Operator

"MgNO’s minimal design, powered by multigrid principles, achieves top-tier PDE predictions while naturally handling boundary conditions and resisting overfitting."

""

**42. PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/19142)\] \[[CODE]()\] 

Tags: PINN, Temporal, Transformer, Wavelets

""

""

**43. SineNet: Learning Temporal Dynamics in Time-Dependent Partial Differential Equations**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18865)\] \[[CODE]()\] 

Tags: Temporal

""

""

**44. Solving High Frequency and Multi-Scale PDEs with Gaussian Processes**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/17752)\] \[[CODE]()\] 

Tags: Gaussian Processes, High-Frequency, Multi-Scale

""

""

**45. Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains**: \[[ICML2024](https://proceedings.mlr.press/v235/lingsch24a.html)\] \[[CODE]()\] 

Tags: Arbitrary Domain, Neural Operator, Spectral Transform

"By employing a truncated direct spectral transform, this work generalizes Fourier neural operators to arbitrary geometries without sacrificing efficiency or accuracy."

""

**46. DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training**: \[[ICML2024](https://proceedings.mlr.press/v235/hao24d.html)\] \[[CODE]()\] 

Tags: Neural Operator, Pre-Training, Transformer

"DPOT merges a denoising auto-regressive pre-training strategy with a scalable Fourier transformer, unlocking robust, large-scale PDE operator learning."

""

**47. Equivariant Graph Neural Operator for Modeling 3D Dynamics**: \[[ICML2024](https://proceedings.mlr.press/v235/xu24j.html)\] \[[CODE]()\] 

Tags: 3D, Neural Operator, Temporal

"EGNO unifies operator learning with 3D-equivariant temporal convolutions to predict entire solution trajectories, surpassing single-step approaches in modeling complex 3D dynamics."

""

**48. Graph Neural PDE Solvers with Conservation and Similarity-Equivariance**: \[[ICML2024](https://proceedings.mlr.press/v235/horie24a.html)\] \[[CODE]()\] 

Tags: Conservation, GNN

"FluxGNNs embed local conservation and similarity symmetry into a graph-based framework, achieving high fidelity and robust generalization for PDEs on diverse domains."

""

**49. HAMLET: Graph Transformer Neural Operator for Partial Differential Equations**: \[[ICML2024](https://proceedings.mlr.press/v235/bryutkin24a.html)\] \[[CODE]()\] 

Tags: Arbitrary Domain, Graph, Neural Operator, Transformer

"HAMLET integrates graph transformers and modular encoders to flexibly solve PDEs on arbitrary geometries with enhanced robustness and efficiency."

""

**50. Harnessing the Power of Neural Operators with Automatically Encoded Conservation Laws**: \[[ICML2024](https://proceedings.mlr.press/v235/liu24p.html)\] \[[CODE]()\] 

Tags: Conservation, Divergence, Neural Operator

"clawNO integrates fundamental conservation laws into its neural operator design, yielding robust, physically consistent solutions even with sparse or noisy data."

""

**51. Hierarchical Neural Operator Transformer with Learnable Frequency-aware Loss Prior for Arbitrary-scale Super-resolution**: \[[ICML2024](https://proceedings.mlr.press/v235/luo24g.html)\] \[[CODE]()\] 

Tags: Multi-Scale, Neural Operator, Super-Resolution, Transformer

"This hierarchical neural operator integrates Galerkin self-attention with a frequency-aware loss prior, achieving resolution-invariant super-resolution and outperforming existing methods across various scientific tasks."

""

**52. Improved Operator Learning by Orthogonal Attention**: \[[ICML2024](https://proceedings.mlr.press/v235/xiao24c.html)\] \[[CODE]()\] 

Tags: Neural Operator, Regularization, Transformer

"By incorporating an orthogonal attention mechanism, ONO achieves strong regularization and improves accuracy in neural operator tasks, reducing overfitting and outperforming baselines."

""

**53. Neural operators meet conjugate gradients: The FCG-NO method for efficient PDE solving**: \[[ICML2024](https://proceedings.mlr.press/v235/rudikov24a.html)\] \[[CODE]()\] 

Tags: Hybrid, Krylov, Neural Operator, Precondition

"FCG-NO bridges neural operator learning with classical iterative solvers to deliver efficient, resolution-invariant PDE preconditioning via energy norm-driven training."

""

**54. Neural Operators with Localized Integral and Differential Kernels**: \[[ICML2024](https://proceedings.mlr.press/v235/liu-schiaffini24a.html)\] \[[CODE]()\] 

Tags: Multi-Resolution, Neural Operator

"By leveraging localized differential and integral kernels, this approach rectifies the global smoothing issue in FNO, achieving substantial accuracy gains across diverse PDE tasks."

""

**55. Parameterized Physics-informed Neural Networks for Parameterized PDEs**: \[[ICML2024](https://proceedings.mlr.press/v235/cho24b.html)\] \[[CODE]()\] 

Tags: PINN

"P2INNs enhance PINNs by embedding parameter representations, enabling a single, robust model for parameterized PDEs and significantly boosting accuracy."

""

**56. Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning**: \[[ICML2024](https://proceedings.mlr.press/v235/chen24au.html)\] \[[CODE]()\] 

Tags: Neural Operator, Transformer

"By emphasizing spatial interrelations over raw function values, PiT offers a more interpretable, efficient attention mechanism that excels in operator learning across multiple PDE tasks."

""

**57. Reference Neural Operators: Learning the Smooth Dependence of Solutions of PDEs on Geometric Deformations**: \[[ICML2024](https://proceedings.mlr.press/v235/cheng24c.html)\] \[[CODE]()\] 

Tags: Efficiency, Geometry, Neural Operator

"RNO leverages a reference geometry and distance-based cross attention to efficiently learn the smooth dependence of PDE solutions on geometric deformations with minimal data."

""

**58. TENG: Time-Evolving Natural Gradient for Solving PDEs With Deep Neural Nets Toward Machine Precision**: \[[ICML2024](https://proceedings.mlr.press/v235/chen24ad.html)\] \[[CODE]()\] 

Tags: Precision, Temporal

"By merging time-dependent variational principles with natural gradient optimization, TENG attains near machine-precision PDE solutions and surpasses state-of-the-art baselines."

""

**59. Transolver: A Fast Transformer Solver for PDEs on General Geometries**: \[[ICML2024](https://proceedings.mlr.press/v235/wu24r.html)\] \[[CODE]()\] 

Tags: Geometry, Transformer

"Transolver’s physics-driven tokenization and attention yield state-of-the-art PDE solutions on complex geometries with enhanced efficiency and scalability."

""

**60. UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs**: \[[ICML2024](https://proceedings.mlr.press/v235/han24a.html)\] \[[CODE]()\] 

Tags: Convergence Guarantee, Multigrid

"UGrid fuses U-Net and multi-grid techniques under a mathematically rigorous framework, guaranteeing convergence, accuracy, and robust self-supervised PDE solving."

""

**61. Vectorized Conditional Neural Fields: A Framework for Solving Time-dependent Parametric Partial Differential Equations**: \[[ICML2024](https://icml.cc/virtual/2024/poster/32919)\] \[[CODE]()\] 

Tags: Neural Fields, Super-Resolution, Temporal, Transformer

"VCNeFs vectorize multiple spatiotemporal queries and condition on PDE parameters, unifying neural fields with attention to enable continuous solutions, zero-shot super-resolution, and robust generalization."

""

**62. Alias-Free Mamba Neural Operator**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/94139)\] \[[CODE](https://github.com/ZhengJianwei2/Mamba-Neural-Operator)\] 

Tags: Alias-Free, Mamba, Neural Operator, State-Space

"Mamba Neural Operator (MambaNO) introduces an alias-free state-space model for PDE solving, achieving state-of-the-art accuracy with O(N) complexity, fewer parameters, and superior efficiency."

"Mamba Neural Operator (MambaNO) 引入无混叠状态空间模型 (SSM) 以求解 PDE，在 O(N) 计算复杂度下实现 SOTA 精度，并显著减少参数量和计算开销。"

**63. AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/96233)\] \[[CODE](https://github.com/LouisSerrano/aroma)\] 

Tags: Neural Fields, Transformer

"AROMA introduces a latent neural field framework with local attention, enabling structure-preserving and efficient PDE modeling across diverse geometries while improving long-term stability with diffusion-based training."

"AROMA 提出基于局部神经场的潜在表示框架，结合 注意力机制 以 保持空间结构 并 高效建模 PDE，通过 扩散训练 提升长时间预测稳定性。"

**64. DiffusionPDE: Generative PDE-Solving under Partial Observation**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/92980)\] \[[CODE](https://jhhuangchloe.github.io/Diffusion-PDE/)\] 

Tags: Diffusion, Inverse

"DiffusionPDE introduces a generative approach for PDE solving under partial observation, leveraging diffusion models to reconstruct missing information and solve PDEs simultaneously."

"DiffusionPDE 提出了一种基于扩散模型的生成式 PDE 求解方法，在部分观测场景下 同时补全缺失信息并求解 PDE，显著提升了正问题和逆问题的求解能力。"

**65. Dual Cone Gradient Descent for Training Physics-Informed Neural Networks**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/94109)\] \[[CODE](https://github.com/youngsikhwang/Dual-Cone-Gradient-Descent)\] 

Tags: Optimization, PINN, Training

"This work identifies gradient imbalance issues in training PINNs and introduces Dual Cone Gradient Descent (DCGD), a novel optimization method that ensures balanced updates, leading to improved stability and accuracy."

"该研究揭示了 PINNs 训练中的梯度失衡问题，并提出双锥梯度下降 (DCGD) 优化方法，以确保梯度更新的平衡性，从而提升稳定性和准确性。"

**66. Fourier Neural Operator with Learned Deformations for PDEs on General Geometries**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/98327)\] \[[CODE]()\] 

Tags: Efficiency, Geometry, Neural Operator

"Geo-FNO introduces a geometry-aware Fourier neural operator that learns to deform irregular domains into a uniform latent space, significantly improving efficiency and accuracy in solving PDEs on arbitrary geometries."

"Geo-FNO 提出了一种几何感知的 Fourier 神经算子，通过学习变换 将非规则域映射到均匀网格，在 任意几何结构的 PDE 求解 任务中 大幅提升计算效率和精度。"

**67. FUSE: Fast Unified Simulation and Estimation for PDEs**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/94332)\] \[[CODE](https://github.com/camlab-ethz/FUSE)\] 

Tags: Efficiency, Inverse, Neural Operator

"FUSE introduces a unified framework for forward and inverse PDE problems, leveraging Fourier Neural Operators and probabilistic estimation to improve both simulation accuracy and parameter inference efficiency."

"FUSE 提出了一个统一的 PDE 正逆问题求解框架，结合 Fourier 神经算子和概率推断，同时提升 物理场预测 和 参数估计 的准确性与计算效率。"

**68. Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93933)\] \[[CODE]()\] 

Tags: Efficiency, Optimization, PINN

"This paper introduces a KFAC-based optimization method to improve the scalability of PINNs, significantly reducing computational costs while maintaining high accuracy in solving PDEs."

"该研究提出了一种基于 KFAC 的优化方法，提高 PINNs 训练的可扩展性，大幅降低计算成本，同时保持高精度求解 PDEs。"

**69. Latent Neural Operator for Solving Forward and Inverse PDE Problems**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/94908)\] \[[CODE](https://github.com/L-I-M-I-T/LatentNeuralOperator)\] 

Tags: Efficiency, Inverse, Neural Operator, Transformer

"LNO introduces a latent-space neural operator with a Physics-Cross-Attention mechanism, significantly improving efficiency and accuracy in both forward and inverse PDE problems."

"LNO 通过 Physics-Cross-Attention 机制 在隐空间学习神经算子，显著提升 PDE 正问题和逆问题的 计算效率与预测精度。"

**70. Multiple Physics Pretraining for Spatiotemporal Surrogate Models**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/96095)\] \[[CODE](https://github.com/PolymathicAI/multiple_physics_pretraining)\] 

Tags: Multi-Physics, Pre-Training, Temporal, Transfer Learning

"MPP introduces a task-agnostic pretraining framework for spatiotemporal surrogate models, enabling broad generalization across diverse physical systems and improving transfer learning."

"MPP 提出了一种物理代理模型的无任务特定预训练框架，实现跨多物理系统的泛化，并提高 迁移学习 能力。"

**71. Neural Krylov Iteration for Accelerating Linear System Solving**: \[[NeurIPS2024](https://neurips.cc/virtual/2024/poster/94379)\] \[[CODE](https://github.com/smart-JLuo/NeurKItt)\] 

Tags: Hybrid, Krylov, Neural Operator

"This paper introduces NeurKItt, a neural operator-based method that accelerates Krylov iteration by predicting the invariant subspace of linear systems, significantly reducing computational cost and iterations."

"本文提出 NeurKItt，一种基于神经算子的 Krylov 迭代加速方法，通过预测线性系统的不变子空间，显著减少计算成本和迭代次数。"

**72. Newton Informed Neural Operator for Solving Nonlinear Partial Differential Equations**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95997)\] \[[CODE]()\] 

Tags: Neural Operator, Newton Methods

"NINO integrates Newton’s method with operator learning to efficiently solve nonlinear PDEs with multiple solutions, significantly reducing computational costs."

"NINO 结合 Newton 方法与算子学习，高效求解具有多个解的非线性 PDE，并 显著降低计算成本。"

**73. On conditional diffusion models for PDE simulations**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93694)\] \[[CODE](https://github.com/cambridge-mlg/pdediff)\] 

Tags: Diffusion

"This work enhances conditional diffusion models for PDE forecasting and data assimilation, introducing autoregressive sampling and novel training strategies for robust performance."

"本文改进了条件扩散模型在 PDE 预测和数据同化中的表现，引入自回归采样和新训练策略，以提高模型的稳定性和泛化能力。"

**74. P2C2Net: PDE-Preserved Coarse Correction Network for efficient prediction of spatiotemporal dynamics**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93729)\] \[[CODE](https://github.com/intell-sci-comput/P2C2Net)\] 

Tags: Temporal

"This work introduces P2C2Net, a physics-encoded correction learning model that efficiently predicts spatiotemporal PDE dynamics on coarse grids with minimal training data."

"本文提出 P2C2Net，一种物理编码的修正学习模型，能够在粗网格和小数据条件下高效预测时空 PDE 动力学。"

**75. Physics-informed Neural Networks for Functional Differential Equations: Cylindrical Approximation and Its Convergence Guarantees**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95857)\] \[[CODE](https://github.com/TaikiMiyagawa/FunctionalPINN)\] 

Tags: Convergence Guarantee, High-dimensional, PINN

"This paper introduces a PINN-based framework for solving Functional Differential Equations (FDEs) using cylindrical approximation, providing convergence guarantees and improving computational efficiency."

"该研究提出了一种基于 PINN 的求解 FDEs 的框架，结合柱面逼近方法，提供收敛性保证，并提升计算效率。"

**76. Physics-Informed Variational State-Space Gaussian Processes**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93352)\] \[[CODE](https://github.com/ohamelijnck/physs_gp)\] 

Tags: Gaussian Processes, Temporal, Variational Inference

"This work introduces PHYSS-GP, a physics-informed state-space Gaussian Process that efficiently handles linear and nonlinear PDEs while maintaining linear-in-time complexity."

"本文提出 PHYSS-GP，一种结合物理先验的状态空间高斯过程方法，能够高效处理线性和非线性 PDE，并保持 线性时间复杂度。"

**77. Poseidon: Efficient Foundation Models for PDEs**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95731)\] \[[CODE](https://github.com/camlab-ethz/poseidon)\] 

Tags: Efficiency, Foundation Model, Multi-Scale, Neural Operator, Transformer

"POSEIDON is a scalable foundation model for PDEs, leveraging a multiscale operator Transformer and semi-group-based training, achieving strong generalization across unseen physical processes."

"POSEIDON 是一个可扩展的 PDE 基础模型，采用多尺度算子 Transformer 和基于半群的训练策略，在未见物理过程上展现出强大的泛化能力。"

**78. Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93155)\] \[[CODE](https://github.com/neuraloperator/CoDA-NO)\] 

Tags: Multi-Physics, Neural Operator, Pre-Training, Transformer

"CoDA-NO introduces codomain attention into neural operators, enabling a self-supervised foundation model for multiphysics PDEs, achieving strong generalization across diverse physical systems."

"CoDA-NO 通过通道注意力机制构建神经算子，实现多物理场 PDE 的自监督基础模型，在不同物理系统间展现出强泛化能力。"

**79. RandNet-Parareal: a time-parallel PDE solver using Random Neural Networks**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/96326)\] \[[CODE](https://github.com/Parallel-in-Time-Differential-Equations/RandNet-Parareal)\] 

Tags: Temporal, Time-Parallel

"RandNet-Parareal integrates random neural networks into time-parallel PDE solvers, significantly improving efficiency and scalability for complex spatiotemporal systems."

"RandNet-Parareal 结合随机神经网络和时间并行求解器，大幅提升时空耦合 PDE 系统的计算效率和可扩展性。"

**80. RoPINN: Region Optimized Physics-Informed Neural Networks**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93144)\] \[[CODE](https://github.com/thuml/RoPINN)\] 

Tags: PINN

"RoPINN enhances PINN optimization by introducing region-based training, improving generalization and high-order constraint satisfaction without extra gradient computation."

"RoPINN 通过区域优化提升 PINN 训练，增强泛化能力和高阶约束满足性，无需额外梯度计算。"

**81. Space-Time Continuous PDE Forecasting using Equivariant Neural Fields**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93158)\] \[[CODE]()\] 

Tags: Meta-Learning, Neural Fields, Temporal

"This work introduces an equivariant neural field framework for space-time continuous PDE forecasting, improving generalization and data efficiency by enforcing PDE symmetries."

"本文提出了一种空间-时间连续的等变神经场框架，通过引入 PDE 对称性约束，提高泛化能力和数据效率。"

**82. Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93621)\] \[[CODE](https://ml-jku.github.io/UPT)\] 

Tags: Neural Operator, Transformer

"This work introduces Universal Physics Transformers (UPTs), a unified and scalable neural operator that efficiently handles diverse spatiotemporal PDE simulations across different grid and particle representations."

"本文提出了 Universal Physics Transformers (UPTs)，一种统一且可扩展的神经算子，能够高效处理不同网格和粒子表示的时空 PDE 仿真问题。"

**83. A physics-informed diffusion model for high-fidelity flow field reconstruction**: \[[Journal of Computational Physics2023](https://www.sciencedirect.com/science/article/pii/S0021999123000670)\] \[[CODE]()\] 

Tags: Diffusion, Fluid

""

""

**84. A Stable and Scalable Method for Solving Initial Value PDEs with Neural Networks**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/10864)\] \[[CODE](https://github.com/mfinzi/neural-ivp)\] 

Tags: IVPs

"Neural-IVP improves ODE-based methods for solving IVPs by addressing numerical instability and computational scalability, enabling efficient and stable neural network evolution of complex PDE dynamics."

"Neural-IVP 通过优化 ODE 方法，解决了初值问题 (IVP) 求解中的数值不稳定性和计算扩展性问题，使神经网络能够高效、稳定地演化复杂 PDE 动力学。"

**85. Clifford Neural Layers for PDE Modeling**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11825)\] \[[CODE](https://microsoft.github.io/cliffordlayers/)\] 

Tags: Geometry, Neural Operator

"Clifford neural layers introduce multivector fields and Clifford algebra to enable geometrically consistent convolutions and Fourier transforms, enhancing the generalization of neural PDE surrogates."

"Clifford 神经层引入多向量场和 Clifford 代数，实现几何一致的卷积与傅里叶变换，提高神经 PDE 代理的泛化能力。"

**86. Competitive Physics Informed Networks**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/10870)\] \[[CODE]()\] 

Tags: PINN, Training

"CPINNs introduce adversarial training to adaptively learn error distributions, achieving four orders of magnitude higher accuracy than traditional PINNs."

"CPINNs 通过引入对抗训练，使物理约束网络能够自适应学习误差分布，实现比传统 PINNs 高四个数量级的精度。"

**87. Continuous PDE Dynamics Forecasting with Implicit Neural Representations**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/12133)\] \[[CODE]()\] 

Tags: Neural Fields

"DINO separates space and time modeling using Implicit Neural Representations (INRs) and a learned ODE, enabling PDE forecasting at arbitrary spatiotemporal locations with improved generalization."

"DINO 通过隐式神经表示（INRs）和学习的常微分方程（ODE）分离空间和时间建模，实现任意时空点的 PDE 预测，提升泛化能力。"

**88. Coupled Multiwavelet Operator Learning for Coupled Differential Equations**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/10753)\] \[[CODE]()\] 

Tags: Neural Operator, Wavelets

"CMWNO decouples coupled PDEs via multiwavelet transform, learning and reconstructing integral kernels in the Wavelet space for high-accuracy solutions."

"CMWNO 通过多小波变换解耦耦合偏微分方程（PDEs），在 Wavelet 空间中学习和重构积分核，实现高精度求解耦合系统。"

**89. CROM: Continuous Reduced-Order Modeling of PDEs Using Implicit Neural Representations**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/12094)\] \[[CODE](https://crom-pde.github.io/)\] 

Tags: Neural Fields

"CROM leverages implicit neural representations (INRs) for discretization-independent reduced-order modeling, enhancing PDE-solving efficiency and adaptability."

"CROM 采用隐式神经表示 (INR) 进行离散化无关的降阶建模，提高 PDE 求解的计算效率和适应性。"

**90. Factorized Fourier Neural Operators**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/10680)\] \[[CODE](https://github.com/alasdairtran/fourierflow)\] 

Tags: Efficiency, Neural Operator

"F-FNO introduces a factorized Fourier representation with improved residual connections, significantly improving the efficiency and accuracy of Fourier-based neural operators for PDE simulation."

"F-FNO 引入分解的傅里叶表示和改进的残差连接，大幅提升傅里叶神经算子在PDE模拟中的效率和精度。"

**91. Guiding continuous operator learning through Physics-based boundary constraints**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11910)\] \[[CODE]()\] 

Tags: Boundary, Neural Operator

"This paper introduces BOON, a boundary-enforcing operator network that structurally modifies neural operators to ensure satisfaction of physics-based boundary conditions, significantly improving solution accuracy."

"本文提出 BOON，一种边界约束神经算子，通过对算子核进行结构性修改，使得 PDE 计算结果严格满足物理边界条件，从而显著提高解的精度。"

**92. Learning Controllable Adaptive Simulation for Multi-resolution Physics**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11428)\] \[[CODE](http://snap.stanford.edu/lamp)\] 

Tags: GNN, RL

"This paper proposes LAMP, a deep learning-based framework that jointly optimizes PDE simulation accuracy and computational cost through adaptive spatial resolution refinement."

"本文提出 LAMP，一个基于深度学习的 自适应多分辨率 PDE 模拟框架，通过学习空间分辨率优化策略 提升 PDE 计算效率并降低预测误差。"

**93. A Neural PDE Solver with Temporal Stencil Modeling**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24181)\] \[[CODE](https://github.com/Edward-Sun/TSM-PDE)\] 

Tags: Fluid, Temporal

"This paper introduces Temporal Stencil Modeling (TSM), a neural PDE solver that enhances spatio-temporal resolution recovery for turbulent flows, achieving state-of-the-art accuracy and efficiency."

"本文提出 时间模板建模 (TSM)，一种基于神经网络的 PDE 求解器，可恢复湍流模拟中的时空细节，显著提升计算精度和效率。"

**94. Gaussian Process Priors for Systems of Linear Partial Differential Equations with Constant Coefficients**: \[[ICML2023](https://icml.cc/virtual/2023/poster/25103)\] \[[CODE](https://github.com/haerski/EPGP)\] 

Tags: Gaussian Processes

"This paper presents EPGP, a Gaussian Process prior framework for solving linear PDEs with constant coefficients, offering a principled, scalable alternative to PINNs."

"本文提出 EPGP, 一种基于高斯过程先验的 线性常系数 PDE 求解框架，可作为 PINN 的高效替代方案。"

**95. Geometric Clifford Algebra Networks**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24098)\] \[[CODE]()\] 

Tags: Fluid, Geometry

"This paper introduces GCANs, a neural network architecture leveraging geometric Clifford algebra for improved modeling of rigid body transformations and fluid dynamics."

"本文提出 GCANs, 一种利用 Clifford 几何代数 进行 刚体变换和流体动力学建模 的神经网络架构。"

**96. GNOT: A General Neural Operator Transformer for Operator Learning**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23985)\] \[[CODE](https://github.com/thu-ml/GNOT)\] 

Tags: Geometry, Multi-Scale, Neural Operator, Transformer

"This paper introduces GNOT, a scalable Transformer-based neural operator designed to handle irregular meshes, multiple input functions, and multi-scale PDE problems, significantly improving operator learning efficiency."

"本文提出 GNOT，一种可扩展的 基于 Transformer 的神经算子，能够处理 不规则网格、多输入函数和多尺度 PDE 问题，在算子学习任务上显著提升效率。"

**97. Group Equivariant Fourier Neural Operators for Partial Differential Equations**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23875)\] \[[CODE]()\] 

Tags: Neural Operator

"This paper proposes G-FNO, a group equivariant Fourier Neural Operator that efficiently leverages symmetries in the frequency domain, leading to improved generalization and accuracy for PDE solutions across different resolutions."

"本文提出 G-FNO，一种具有 群等变性 的 傅里叶神经算子，在频域中利用对称性信息，从而提升 PDE 求解的泛化能力和精度，尤其在不同离散化尺度下依然表现稳定。"

**98. Implicit Neural Spatial Representations for Time-dependent PDEs**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24695)\] \[[CODE]()\] 

Tags: Neural Fields, Temporal

"This paper introduces INSR, an implicit neural representation for time-dependent PDEs, storing spatial information directly in network weights and evolving via time integration, leading to improved accuracy, memory efficiency, and adaptivity."

"本文提出 INSR，一种用于时间依赖型 PDE 的 隐式神经空间表示，在网络权重中存储空间信息，并通过时间积分更新，实现更高精度、更低内存占用和自适应求解能力。"

**99. Learning Neural PDE Solvers with Parameter-Guided Channel Attention**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24567)\] \[[CODE](https://github.com/nec-research/CAPE-ML4Sci)\] 

Tags: Autoregressive

"This paper introduces CAPE, a parameter-guided channel attention module that enhances neural PDE solvers by improving generalization to unseen PDE parameters and seamlessly integrating autoregressive learning."

"本文提出 CAPE，一种 参数引导的通道注意力机制，能够增强神经 PDE 求解器，使其更好地适应 未见过的 PDE 参数，并结合自回归学习策略以提高泛化能力。"

**100. Learning Preconditioners for Conjugate Gradient PDE Solvers**: \[[ICML2023](https://icml.cc/virtual/2023/poster/25127)\] \[[CODE](https://sites.google.com/view/neuralPCG)\] 

Tags: GNN, Precondition

"This paper introduces a learning-based preconditioner for conjugate gradient PDE solvers, leveraging graph neural networks and a novel loss function to improve efficiency and generalizability."

"本文提出了一种 基于学习的共轭梯度 PDE 求解预条件子，利用 图神经网络（GNN） 和新型损失函数，提高求解器的 效率和泛化能力。"

**101. Meta Learning of Interface Conditions for Multi-Domain Physics-Informed Neural Networks**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24902)\] \[[CODE]()\] 

Tags: Gaussian Processes, Meta-Learning, PINN

"This paper introduces META Learning of Interface Conditions (METALIC), a contextual bandit-based approach to optimize interface conditions in multi-domain PINNs, significantly improving solution accuracy for parametric PDEs."

"本文提出了一种 基于上下文多臂赌博（MAB）的界面条件优化方法 METALIC，用于 多域 PINNs，能够 自适应选择最优界面条件，提高 PDE 求解精度。"

**102. MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23616)\] \[[CODE]()\] 

Tags: GNN, Multigrid, Precondition

"This paper proposes MG-GNN, a novel multigrid graph neural network to optimize two-level domain decomposition methods (DDMs), achieving superior generalization to large-scale unstructured grids."

"本文提出了一种 多重网格图神经网络（MG-GNN），用于优化 二层域分解方法（DDMs），可有效泛化到大规模非结构网格 PDE 求解。"

**103. NeuralStagger: Accelerating Physics-constrained Neural PDE Solver with Spatial-temporal Decomposition**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23962)\] \[[CODE]()\] 

Tags: Efficiency, Temporal

"This paper introduces NeuralStagger, a spatial-temporal decomposition method that parallelizes low-resolution neural PDE solvers, achieving 10∼100× speed-up while preserving accuracy and flexibility."

"本文提出 NeuralStagger，通过 空间-时间分解 并行训练 低分辨率神经 PDE 求解器，在保证准确性的同时 加速 10∼100 倍 并提供 计算资源-分辨率可调性。"

**104. NUNO: A General Framework for Learning Parametric PDEs with Non-Uniform Data**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23746)\] \[[CODE](https://github.com/thu-ml/NUNO)\] 

Tags: Neural Operator, Non-Uniform

"This paper introduces NUNO, a non-uniform neural operator framework that leverages K-D tree domain decomposition to efficiently learn PDEs from non-uniform data, achieving significant error reduction (34%–61%) and speedup (2×–30×)."

"本文提出 NUNO 框架，通过 K-D 树域分解 解决 非均匀数据 PDE 学习问题，实现 误差降低 (34%–61%)，训练加速 (2×–30×)。"

**105. Q-Flow: Generative Modeling for Differential Equations of Open Quantum Dynamics with Normalizing Flows**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23549)\] \[[CODE]()\] 

Tags: Normalizing Flow, Quantum Dynamics

"This paper introduces Q-Flow, a generative modeling approach for open quantum dynamics PDEs using normalizing flows, demonstrating superior accuracy and efficiency over classical and ML-based PDE solvers."

"本文提出 Q-Flow，一种基于 归一化流 的生成建模方法，针对 开放量子系统 PDE，在 经典和 ML PDE 求解器 之上实现更高效、更准确的求解。"

**106. Random Grid Neural Processes for Parametric Partial Differential Equations**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24951)\] \[[CODE]()\] 

Tags: Gaussian Processes, Inverse, Variational Inference

"This paper introduces Random Grid Neural Processes, a probabilistic learning framework for solving parametric PDEs, enabling grid-invariant neural modeling and efficient forward/inverse problem solving."

"本文提出 随机网格神经过程，一种 参数化 PDE 的概率学习框架，实现了 网格不变的神经建模和高效的正问题/逆问题求解。"

**107. Solving High-Dimensional PDEs with Latent Spectral Models**: \[[ICML2023](https://icml.cc/virtual/2023/poster/25165)\] \[[CODE](https://github.com/thuml/Latent-Spectral-Models)\] 

Tags: High-dimensional

"This paper introduces Latent Spectral Models (LSM), leveraging hierarchical projection networks and neural spectral blocks to efficiently solve high-dimensional PDEs in a compact latent space."

"本文提出 潜在谱模型 (LSM)，利用 层次投影网络和神经谱块 在紧凑的潜在空间中高效求解高维 PDE。"

**108. Convolutional Neural Operators for robust and accurate learning of PDEs**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71891)\] \[[CODE](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator)\] 

Tags: Neural Operator

"本文提出卷积神经算子 (CNO)，将 CNN 结构适配于 PDE 解算子学习，实现稳健的函数空间建模，并具备良好的泛化能力。"

"This paper introduces Convolutional Neural Operators (CNOs), a novel adaptation of CNNs for learning PDE solution operators, ensuring robust function-space learning with strong generalization capabilities."

**109. Deep Equilibrium Based Neural Operators for Steady-State PDEs**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/70120)\] \[[CODE]()\] 

Tags: Neural Operator

"This paper introduces FNO-DEQ, a weight-tied deep equilibrium variant of Fourier Neural Operator (FNO) for solving steady-state PDEs, achieving superior accuracy with lower memory requirements and demonstrating robustness against noisy observations."

"本文提出 FNO-DEQ，一种基于权重共享的深度平衡 Fourier 神经算子，用于求解稳态 PDE，具有更低的内存占用、更高的精度，并在噪声数据下展现出更强的鲁棒性。"

**110. Domain Agnostic Fourier Neural Operators**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/70145)\] \[[CODE](https://github.com/ningliu-iga/DAFNO)\] 

Tags: Boundary, Geometry, Neural Operator

"This paper introduces DAFNO, an extension of Fourier Neural Operators that incorporates explicit domain boundary information, enabling accurate learning on irregular geometries and topology changes while preserving computational efficiency."

"本文提出 DAFNO，一种扩展的 Fourier 神经算子，显式嵌入域边界信息，使其在不规则几何和拓扑变化情况下仍能精准学习，同时保持计算效率。"

**111. Entropy-dissipation Informed Neural Network for McKean-Vlasov Type PDEs**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72246)\] \[[CODE]()\] 

Tags: PINN

"This paper introduces EINN, an entropy-dissipation informed neural network framework for solving McKean-Vlasov equations with singular interaction kernels, ensuring theoretical guarantees and superior empirical performance."

"本文提出 EINN，一种基于熵耗散的物理约束神经网络，用于求解具有奇异相互作用核的 McKean-Vlasov 方程，并提供理论保证及优越的实验表现。"

**112. Equivariant Neural Operator Learning with Graphon Convolution**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72298)\] \[[CODE](https://github.com/ccr-cheng/InfGCN-pytorch)\] 

Tags: Neural Operator

"This paper introduces an SE(3)-equivariant neural operator based on Graphon Convolution (InfGCN), effectively capturing geometric information while ensuring equivariance for learning mappings in 3D Euclidean space."

"本文提出一种基于 Graphon 卷积 (InfGCN) 的 SE(3) 等变神经算子，在 3D 欧几里得空间中学习映射时能有效捕捉几何信息，并确保等变性。"

**113. Geometry-Informed Neural Operator for Large-Scale 3D PDEs**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72670)\] \[[CODE]()\] 

Tags: 3D, Geometry, Neural Operator

"This paper presents GINO, a geometry-informed neural operator that integrates graph and Fourier architectures for solving large-scale 3D PDEs, achieving remarkable efficiency and generalizability in CFD simulations."

"本文提出 GINO，一种几何感知神经算子，结合图神经算子 (GNO) 和傅里叶神经算子 (FNO) 以求解大规模 3D PDE，在 CFD 模拟中展现了卓越的计算效率和泛化能力。"

**114. Learning Space-Time Continuous Latent Neural PDEs from Partially Observed States**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72590)\] \[[CODE](https://github.com/yakovlev31/LatentNeuralPDEs)\] 

Tags: Temporal, Variational Inference

"This paper introduces a space-time continuous latent neural PDE model that effectively learns PDE dynamics from noisy and partially observed data, achieving state-of-the-art performance in grid-independent learning."

"本文提出了一种空间-时间连续的潜变量神经 PDE 模型，可从部分观测数据中学习 PDE 动力学，实现网格无关的高效建模，并在多个任务上优于现有方法。"

**115. Lie Point Symmetry and Physics-Informed Networks**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71137)\] \[[CODE]()\] 

Tags: Lie Algebra, PINN

"This paper introduces a symmetry-informed loss for PINNs based on Lie point symmetries, significantly improving the sample efficiency and generalization of neural PDE solvers."

"本文提出了一种基于 Lie 点对称性的对称性损失，提高了 PINN 在 PDE 求解中的泛化能力和样本效率。"

**116. Nonparametric Boundary Geometry in Physics Informed Deep Learning**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71419)\] \[[CODE]()\] 

Tags: Boundary, Geometry, Neural Operator

"This paper introduces a neural operator architecture that directly incorporates nonparametric boundary geometries, enabling rapid and reusable PDE solutions across different designs."

"本文提出了一种可处理非参数化边界几何的神经算子架构，使得 PDE 解可在不同设计中快速预测并复用。"

**117. Operator Learning with Neural Fields: Tackling PDEs on General Geometries**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72838)\] \[[CODE]()\] 

Tags: Geometry, Neural Fields, Neural Operator

"CORAL introduces a coordinate-based neural field approach for operator learning, enabling PDE solving on general geometries without discretization constraints."

"CORAL 提出了一种基于坐标神经场的神经算子方法，使 PDE 求解摆脱离散化约束，可适用于任意几何结构。"

**118. PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71688)\] \[[CODE](https://github.com/microsoft/pdearena)\] 

Tags: Diffusion, Temporal

"PDE-Refiner leverages multistep refinement inspired by diffusion models to enhance the long-term stability and accuracy of neural PDE solvers, effectively capturing all spatial frequency components."

"PDE-Refiner 受扩散模型启发，通过多步细化增强神经 PDE 求解器的长期稳定性和准确性，有效捕捉全频谱信息。"

**119. Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72712)\] \[[CODE]()\] 

Tags: Alias-Free, Neural Operator

"This work introduces a novel neural operator framework (ReNO) to address aliasing errors caused by discretization in operator learning, ensuring consistency between continuous and discrete representations."

"本文提出了一种新的神经算子框架（ReNO），以解决算子学习中由于离散化引入的混叠误差问题，从而确保连续和离散表示之间的一致性。"

**120. Scalable Transformer for PDE Surrogate Modeling**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71008)\] \[[CODE](https://github.com/BaratiLab/FactFormer)\] 

Tags: High-dimensional, Transformer

"FactFormer introduces a scalable factorized kernel attention mechanism for PDE surrogate modeling, achieving efficient and stable performance in high-dimensional problems."

"FactFormer 提出了一种可扩展的分解核注意力机制，在高维 PDE 代理建模中实现高效且稳定的性能。"

**121. Separable Physics-Informed Neural Networks**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71036)\] \[[CODE](https://jwcho5576.github.io/spinn.github.io/)\] 

Tags: Efficiency, High-dimensional, PINN

"SPINN significantly accelerates PINN training for high-dimensional PDEs by leveraging per-axis processing and forward-mode differentiation, achieving orders of magnitude improvements in speed and efficiency."

"SPINN 通过逐轴计算和前向模式微分，大幅加速高维 PDE 物理引导神经网络的训练，显著提升计算速度和效率。"

**122. Unifying Predictions of Deterministic and Stochastic Physics in Mesh-reduced Space with Sequential Flow Generative Model**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72959)\] \[[CODE]()\] 

Tags: Normalizing Flow

"This work introduces a unified framework combining autoencoders, transformers, and normalizing flows for accurately predicting both deterministic and stochastic PDEs in reduced mesh spaces."

"本文提出了一种融合自编码器、变换器和正规化流的统一框架，可在降维网格空间中精准预测确定性和随机 PDE 动力系统。"

**123. Message Passing Neural PDE Solvers**: \[[ICLR2022](https://iclr.cc/virtual/2022/poster/7134)\] \[[CODE]()\] 

Tags: GNN

"This paper introduces a fully neural PDE solver based on message passing, offering stability and generalization across multiple PDEs and outperforming traditional solvers in low-resolution scenarios."

"本文提出了一种基于神经消息传递的全神经 PDE 求解器，提供了跨多个 PDE 的稳定性和泛化能力，在低分辨率场景中优于传统求解器。"

**124. Predicting Physics in Mesh-reduced Space with Temporal Attention**: \[[ICLR2022](https://iclr.cc/virtual/2022/poster/6494)\] \[[CODE]()\] 

Tags: GNN, Multigrid, Temporal, Transformer

"This paper introduces a hybrid GNN and Transformer model to predict physics in a mesh-reduced space using temporal attention, achieving stable rollouts and outperforming state-of-the-art baselines in complex fluid dynamics."

"本文提出了一种结合 GNN 和 Transformer 的模型，在网格降维空间中通过时间注意力预测物理动态，实现了稳定预测，并在复杂流体动力学任务中优于现有最先进的方法。"

**125. Composing Partial Differential Equations with Physics-Aware Neural Networks**: \[[ICML2022](https://icml.cc/virtual/2022/poster/16235)\] \[[CODE]()\] 

Tags: FVM

"This paper introduces FINN, a compositional physics-aware neural network that integrates finite volume methods with deep learning, demonstrating superior accuracy and generalization in modeling spatiotemporal PDEs, including real-world diffusion-sorption scenarios."

"本文提出了 FINN，一种组合物理感知神经网络，通过结合有限体积方法和深度学习，在时空 PDEs（包括实际扩散-吸附场景）的建模中表现出优越的精度和泛化能力。"

**126. A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53220)\] \[[CODE]()\] 

Tags: Boundary, FEM, Geometry, Hard Constraints

"This paper proposes a unified hard-constraint framework using "extra fields" from the mixed finite element method to stably and efficiently solve geometrically complex PDEs with neural networks, ensuring automatic satisfaction of Dirichlet, Neumann, and Robin boundary conditions."

"本文提出了一种基于混合有限元方法中的“额外场”的统一硬约束框架，在神经网络中稳定且高效地求解几何复杂的偏微分方程 (PDEs)，并自动满足 Dirichlet、Neumann 和 Robin 边界条件。"

**127. Accelerated Training of Physics-Informed Neural Networks (PINNs) using Meshless Discretizations**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53689)\] \[[CODE]()\] 

Tags: PINN

"The paper introduces Discretely-Trained PINNs (DT-PINNs) with meshless radial basis function-finite differences (RBF-FD), achieving 2-4x faster training than traditional PINNs on complex geometries while maintaining accuracy."

"通过引入高阶网格无关离散化方法 (RBF-FD)，显著加速了 PINNs 在复杂几何和高阶导数问题上的训练速度，同时保持精度。"

**128. Gold-standard solutions to the Schrödinger equation using deep learning: How much physics do we need?**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/54906)\] \[[CODE]()\] 

Tags: Schrödinger

"This paper introduces a novel deep learning architecture for solving the Schrödinger equation, achieving a 40-70% reduction in energy error and significantly lower computational costs, with the counter-intuitive finding that excessive physical prior knowledge may hinder optimization."

"本研究提出了一种新型深度学习架构，用于求解薛定谔方程，在降低能量误差40-70%的同时显著降低计算成本，揭示了过多的物理先验知识可能会降低优化效果的反直觉现象。"

**129. Is $L^2$ Physics Informed Loss Always Suitable for Training Physics Informed Neural Network?**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/55104)\] \[[CODE](https://github.com/LithiumDA/L_inf-PINN)\] 

Tags: PINN

"This paper theoretically analyzes the choice of physics-informed loss in PINN training, showing that $L^2$ loss may not be suitable for Hamilton-Jacobi-Bellman (HJB) equations and proposing a novel $L^\infty$ loss-based training algorithm to enhance accuracy."

"本文从理论上探讨了物理损失函数在 PINN 训练中的选择，提出了 $L^2$ 损失在 Hamilton-Jacobi-Bellman (HJB) 方程中可能不适用，推荐采用 $L^\infty$ 损失以提高精度，并提出了一种新的 PINN 训练算法。"

**130. Learning Interface Conditions in Domain Decomposition Solvers**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53135)\] \[[CODE]()\] 

Tags: Domain Decomposition, GNN, Hybrid, Schwarz

"This paper introduces a novel approach using Graph Convolutional Neural Networks (GCNNs) and unsupervised learning to optimize Schwarz domain decomposition methods for solving PDEs on both structured and unstructured grids, significantly improving generalization and computational efficiency."

"本文提出了一种基于图卷积神经网络 (GCNNs) 和无监督学习的方法，优化 Schwarz 域分解方法，在结构化和非结构化网格上求解偏微分方程 (PDEs)，显著提升了泛化能力和计算效率。"

**131. Learning Operators with Coupled Attention**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/56114)\] \[[CODE]()\] 

Tags: Neural Operator, Transformer

"This paper introduces LOCA, a novel operator learning framework with a Kernel-Coupled Attention mechanism, demonstrating strong performance and robustness in solving PDEs and black-box functional relationships with limited data."

"本文提出了 LOCA，一种结合内核耦合注意力机制的算子学习框架，在解决偏微分方程 (PDEs) 和黑箱函数关系时，展现了出色的性能和在数据稀缺情况下的鲁棒性。"

**132. Learning to Accelerate Partial Differential Equations via Latent Global Evolution**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/55007)\] \[[CODE]()\] 

Tags: Efficiency

"This paper proposes LE-PDE, a novel method that accelerates PDE simulations by evolving dynamics in a low-dimensional latent space, achieving significant speedups and maintaining high accuracy."

"本文提出了 LE-PDE，一种通过在低维潜在空间中演化动力学加速PDE仿真的方法，实现了显著的加速效果，同时保持了较高的准确性。"

**133. M2N: Mesh Movement Networks for PDE Solvers**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53649)\] \[[CODE]()\] 

Tags: GNN, Spline, Transformer

"This paper introduces M2N, the first learning-based end-to-end mesh movement framework for PDE solvers, achieving mesh adaptation speedups by 3-4 orders of magnitude while maintaining numerical accuracy."

"本文提出了 M2N，这是首个基于学习的端到端网格移动框架，在保持数值精度的同时，将网格自适应速度提升了 3-4 个数量级。"

**134. MAgNet: Mesh Agnostic Neural PDE Solver**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/52976)\] \[[CODE](https://github.com/jaggbow/magnet)\] 

Tags: GNN, Mesh, Neural Fields, Zero-Shot

"This paper introduces MAgNet, a mesh-agnostic neural PDE solver that leverages implicit neural representations and graph neural networks to achieve zero-shot generalization to new meshes and enable long-term, physically consistent predictions."

"本文提出了 MAgNet，一种网格无关的神经PDE求解器，结合隐式神经表示和图神经网络，实现对新网格的零样本泛化，并能够进行长期物理一致的预测。"

**135. Meta-Auto-Decoder for Solving Parametric Partial Differential Equations**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53638)\] \[[CODE](https://gitee.com/mindspore/mindscience/tree/master/MindElec/)\] 

Tags: Manifold, Mesh, Meta-Learning

"The Meta-Auto-Decoder (MAD) method provides a mesh-free, unsupervised deep learning framework for efficiently solving parametric PDEs by leveraging meta-learning and manifold learning to enable fast adaptation to new equation instances."

"Meta-Auto-Decoder (MAD) 方法通过结合元学习和流形学习，提供了一种网格无关、无监督的深度学习框架，高效求解参数化PDE，并实现对新方程实例的快速适应。"

**136. Neural Stochastic PDEs: Resolution-Invariant Learning of Continuous Spatiotemporal Dynamics**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/54074)\] \[[CODE]()\] 

Tags: Neural Operator

"The Neural SPDE model extends Neural CDEs and Neural Operators by enabling resolution-invariant learning of spatiotemporal dynamics under stochastic influences, demonstrating higher accuracy and faster performance than traditional solvers."

"Neural SPDE 模型通过扩展 Neural CDEs 和 Neural Operators，实现了在随机影响下时空动态的分辨率无关学习，表现出比传统求解器更高的准确性和更快的性能。"

**137. NOMAD: Nonlinear Manifold Decoders for Operator Learning**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53042)\] \[[CODE]()\] 

Tags: Manifold, Neural Operator

"NOMAD introduces a nonlinear manifold decoder for operator learning, enabling efficient low-dimensional representations of solution manifolds in PDEs and achieving high accuracy with reduced model size and training cost."

"NOMAD 引入了一种非线性流形解码器，用于算子学习，实现了PDE解流形的高效低维表示，在模型尺寸和训练成本较小的情况下实现了高精度。"

**138. Physics-Embedded Neural Networks: Graph Neural PDE Solvers with Mixed Boundary Conditions**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/54479)\] \[[CODE](https://github.com/yellowshippo/penn-neurips2022)\] 

Tags: Boundary, GNN

"Physics-Embedded Neural Networks integrate boundary conditions and global connectivity into E(n)-equivariant GNNs, enabling accurate and stable long-term predictions of complex physical systems governed by PDEs."

"Physics-Embedded Neural Networks 将边界条件和全局连接融入 E(n)-等变 GNN 中，实现了复杂物理系统的精确稳定长时间预测。"

**139. Transform Once: Efficient Operator Learning in Frequency Domain**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53622)\] \[[CODE]()\] 

Tags: Frequency Domain, Neural Operator, Temporal

"Transform Once (T1) 提出了一种高效的频域深度学习方法，通过一次频域转换实现快速且精度更高的偏微分方程 (PDE) 解算，显著减少计算成本。"

"Transform Once (T1) introduces an efficient frequency-domain deep learning method that achieves faster and more accurate PDE solutions through a single frequency transformation, significantly reducing computational costs."

**140. Fourier Neural Operator for Parametric Partial Differential Equations**: \[[ICLR2021](https://iclr.cc/virtual/2021/poster/3281)\] \[[CODE]()\] 

Tags: Neural Operator

""

""

**141. Learning continuous-time PDEs from sparse data with graph neural networks**: \[[ICLR2021](https://iclr.cc/virtual/2021/poster/3028)\] \[[CODE]()\] 

Tags: GNN, Temporal

""

""

**142. Solving high-dimensional parabolic PDEs using the tensor train format**: \[[ICML2021](https://icml.cc/virtual/2021/poster/9927)\] \[[CODE]()\] 

Tags: 

""

""

**143. Multiwavelet-based Operator Learning for Differential Equations**: \[[NeurIPS2021](https://neurips.cc/virtual/2021/poster/26769)\] \[[CODE]()\] 

Tags: Neural Operator, Wavelets

""

""

**144. Learning Algebraic Multigrid Using Graph Neural Networks**: \[[ICML2020](https://icml.cc/virtual/2020/poster/6369)\] \[[CODE]()\] 

Tags: GNN, Multigrid

""

""

**145. Multipole Graph Neural Operator for Parametric Partial Differential Equations**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_4b21cf96d4cf612f239a6c322b10c8fe.html)\] \[[CODE]()\] 

Tags: Graph, Neural Operator

""

""

**146. Numerically Solving Parametric Families of High-Dimensional Kolmogorov Partial Differential Equations via Deep Learning**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_c1714160652ca6408774473810765950.html)\] \[[CODE]()\] 

Tags: 

""

""

**147. Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE-Solvers**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_43e4e6a6f341e00671e123714de019a8.html)\] \[[CODE]()\] 

Tags: 

""

""

**148. Learning Neural PDE Solvers with Convergence Guarantees**: \[[ICLR2019](https://openreview.net/forum?id=rklaWn0qK7)\] \[[CODE]()\] 

Tags: 

""

""

**149. Learning to Optimize Multigrid PDE Solvers**: \[[ICML2019](https://openreview.net/forum?id=SJEDWibdWr)\] \[[CODE]()\] 

Tags: Multigrid

""

""

## 2. Inverse

**1. Physics-Informed Deep Inverse Operator Networks for Solving PDE Inverse Problems**: \[[ICLR2025](https://openreview.net/forum?id=0FxnSZJPmh)\] \[[CODE]()\] 

Tags: Neural Operator, Unsupervised

"PI-DIONs integrate stability estimates into operator learning to solve PDE inverse problems without labeled data, ensuring stable and efficient real-time inference."

"PI-DIONs 在算子学习框架中引入稳定性估计，实现无监督 PDE 逆问题求解，确保稳定高效的实时推理。"

**2. PIED: Physics-Informed Experimental Design for Inverse Problems**: \[[ICLR2025](https://openreview.net/forum?id=w7P92BEsb2)\] \[[CODE]()\] 

Tags: Meta-Learning

"PIED leverages PINNs for experimental design in inverse problems, enabling efficient one-shot optimization of design parameters under limited observation budgets."

"PIED 利用 PINNs 进行逆问题的实验设计，在有限观测预算下，实现实验参数的一次性高效优化。"

**3. CONFIDE: Contextual Finite Difference Modelling of PDEs**: \[[KDD2024](https://dl.acm.org/doi/10.1145/3637528.3671676)\] \[[CODE](https://github.com/orilinial/CONFIDE)\] 

Tags: Text-to-PDE, Zero-Shot

"CONFIDE introduces a novel data-driven PDE inference framework that leverages learned context and finite difference modeling to enable efficient, transferable, and interpretable PDE calibration and prediction."

"CONFIDE 提出了一种数据驱动的 PDE 推导框架，结合 上下文学习与有限差分建模，实现高效、可迁移、可解释的 PDE 校准与预测。"

**4. Bi-level Physics-Informed Neural Networks for PDE Constrained Optimization using Broyden's Hypergradients**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11901)\] \[[CODE]()\] 

Tags: PINN

"Bi-level PINNs leverage the Implicit Function Theorem and Broyden’s method to efficiently compute hypergradients, improving the solving capability of PDE-constrained optimization, especially for complex geometries and nonlinear PDEs."

"Bi-level PINNs 结合隐函数定理与 Broyden 方法，高效计算超梯度，提升 PDE 约束优化问题的求解能力，特别适用于复杂几何和非线性问题。"

**5. Neural Inverse Operators for Solving PDE Inverse Problems**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24713)\] \[[CODE]()\] 

Tags: Inverse, Neural Operator

"This paper introduces Neural Inverse Operators (NIOs), a novel framework that composes DeepONet and FNO to efficiently learn inverse mappings from operators to functions, achieving robust, accurate, and fast solutions for PDE inverse problems."

"本文提出了一种新架构 Neural Inverse Operators (NIOs)，结合 DeepONet 和 FNO，高效学习 从算子到函数的逆映射，在 PDE 逆问题 求解上表现出 强鲁棒性、高精度和极快计算速度。"

**6. PETAL: Physics Emulation Through Averaged Linearizations for Solving Inverse Problems**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72426)\] \[[CODE]()\] 

Tags: Ocean

"PETAL incorporates physics-based linearizations into neural network surrogates for solving nonlinear inverse problems, enhancing physical consistency and solution accuracy."

"PETAL 通过结合物理模型的线性化信息来改进神经网络对非线性反问题的求解，增强了梯度信息的物理一致性，提高了逆问题的求解精度。"

**7. Physics-Driven ML-Based Modelling for Correcting Inverse Estimation**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72691)\] \[[CODE]()\] 

Tags: Hybrid

"GEESE enhances the reliability and physical consistency of ML-based inverse estimation by integrating hybrid surrogate error models and generative optimization techniques."

"GEESE 通过混合代理误差模型和生成式方法优化物理约束下的逆问题求解，提高了机器学习估计的可靠性和物理一致性。"

**8. Solving Inverse Physics Problems with Score Matching**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72971)\] \[[CODE](https://github.com/tum-pbs/SMDP)\] 

Tags: Diffusion

"SMDP integrates physics-based simulation with diffusion models, leveraging score matching for accurate and stable inverse problem solving."

"SMDP 结合物理模拟和扩散模型，通过分数匹配改进逆问题求解，实现高精度和稳定的时间反演。"

**9. Learning to Solve PDE-constrained Inverse Problems with Graph Networks**: \[[ICML2022](https://icml.cc/virtual/2022/poster/16565)\] \[[CODE]()\] 

Tags: GNN

"This paper proposes a method combining Graph Neural Networks (GNNs) and autodecoder priors to achieve efficient and accurate estimation of initial conditions and physical parameters in PDE-constrained inverse problems."

"该论文提出了结合图神经网络 (GNN) 和自解码器 (autodecoder) 的方法，在偏微分方程 (PDE) 约束的逆问题中实现高效、精确的初始条件与物理参数估计。"

**10. Robust SDE-Based Variational Formulations for Solving Linear PDEs via Deep Learning**: \[[ICML2022](https://icml.cc/virtual/2022/poster/16565)\] \[[CODE]()\] 

Tags: GNN

"This paper proposes a novel approach combining graph neural networks (GNNs) with continuous coordinate networks to efficiently solve PDE-constrained inverse problems, achieving significant speedups and improved accuracy over traditional solvers."

"本文提出了一种将图神经网络 (GNNs) 与连续坐标网络相结合的新方法，有效解决了 PDE 约束的逆问题，相比传统求解器显著提升了计算速度和准确性。"

**11. Learning Physics Constrained Dynamics Using Autoencoders**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/53182)\] \[[CODE]()\] 

Tags: Autoencoder, Frequency Domain

"The paper combines physics constraints with autoencoders to efficiently predict system states and physical parameters, demonstrating strong performance in high-frequency data scenarios."

"通过将物理约束与自编码器相结合，实现了系统状态和物理参数的高效预测，特别是在高频数据场景下表现出色。"

**12. Scale-invariant Learning by Physics Inversion**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/54120)\] \[[CODE]()\] 

Tags: Optimization

"The paper introduces a hybrid training method by embedding higher-order physics solvers into neural network pipelines, enabling efficient and stable parameter estimation and optimal control in complex physical systems."

"通过将高阶物理逆问题求解器嵌入神经网络训练流程，实现了在复杂物理系统中高效且稳定的参数估计和优化控制。"

**13. Physics-aware, probabilistic model order reduction with guaranteed stability**: \[[ICLR2021](https://iclr.cc/virtual/2021/poster/2719)\] \[[CODE]()\] 

Tags: 

""

""

**14. Amortized Finite Element Analysis for Fast PDE-Constrained Optimization**: \[[ICML2020](https://icml.cc/virtual/2020/poster/6574)\] \[[CODE]()\] 

Tags: 

""

""

## 3. Discovery

**1. LLM-SR: Scientific Equation Discovery via Programming with Large Language Models**: \[[ICLR2025](https://openreview.net/forum?id=m2nmp8P5in)\] \[[CODE]()\] 

Tags: LLM, Symbolic Regression

"LLM-SR integrates Large Language Models with evolutionary search to enhance scientific equation discovery, outperforming traditional symbolic regression in accuracy and generalization."

"LLM-SR 结合大语言模型与进化搜索，实现更高效的科学方程发现，在准确性与泛化能力上优于传统符号回归方法。"

**2. PhysPDE: Rethinking PDE Discovery and a Physical HYpothesis Selection Benchmark**: \[[ICLR2025](https://openreview.net/forum?id=G3CpBCQwNh)\] \[[CODE]()\] 

Tags: Benchmark, Symbolic Regression

"PhysPDE introduces a physically guided PDE discovery framework, leveraging hypothesis selection to enhance interpretability and scientific consistency in machine learning for physics."

"PhysPDE 提出了一种物理引导的 PDE 发现框架，通过假设选择提高可解释性，使机器学习更科学地发现物理定律。"

**3. TRENDy: Temporal Regression of Effective Nonlinear Dynamics**: \[[ICLR2025](https://openreview.net/forum?id=NvDRvtrGLo)\] \[[CODE]()\] 

Tags: Multi-Scale, Temporal

"TRENDy introduces a robust, equation-free approach to modeling spatiotemporal dynamics, leveraging multiscale filtering and neural ODEs to discover effective dynamics and bifurcations."

"TRENDy 提出了一种稳健的无方程方法，通过多尺度滤波和神经 ODE 捕获时空动力学，并自动检测系统分岔行为。"

**4. Nonlocal Attention Operator: Materializing Hidden Knowledge Towards Interpretable Physics Discovery**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93271)\] \[[CODE](https://github.com/fishmoon1234/NAO)\] 

Tags: Inverse, Neural Operator, Transformer

"NAO introduces an attention-based neural operator for solving forward and inverse PDE problems, improving generalization and interpretability through a learned kernel mapping."

"NAO 提出了一种基于注意力机制的神经算子，用于求解正向和逆向 PDE，利用学习的核映射提升泛化能力和物理可解释性。"

**5. Physics-Guided Discovery of Highly Nonlinear Parametric Partial Differential Equations**: \[[KDD2023](https://dl.acm.org/doi/10.1145/3580305.3599466)\] \[[CODE]()\] 

Tags: 

""

""

**6. Symbolic Physics Learner: Discovering governing equations via Monte Carlo tree search**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11641)\] \[[CODE]()\] 

Tags: Symbolic Regression

"This paper introduces Symbolic Physics Learner (SPL), a Monte Carlo tree search-based method for discovering governing equations of nonlinear dynamical systems, enforcing parsimony and accuracy."

"本文提出了 Symbolic Physics Learner (SPL)，一种基于蒙特卡洛树搜索 (MCTS) 的方法，用于发现非线性动力系统的控制方程，同时保证解析表达式的简洁性和精确性。"

**7. Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics**: \[[ICML2023](https://icml.cc/virtual/2023/poster/25243)\] \[[CODE](https://sites.google.com/view/nclaw)\] 

Tags: Hybrid, PINN, Zero-Shot

"This paper introduces Neural Constitutive Laws (NCLaw), a hybrid NN-PDE framework that explicitly incorporates governing PDEs while learning constitutive models, achieving superior generalization across geometries, boundary conditions, and multi-physics settings."

"本文提出 Neural Constitutive Laws (NCLaw)，一种结合 NN 和 PDE 的混合框架，显式利用已知 PDE 结构，仅学习材料本构关系，实现对几何、边界条件和多物理系统的强泛化能力。"

**8. Universal Physics-Informed Neural Networks: Symbolic Differential Operator Discovery with Sparse Data**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23514)\] \[[CODE]()\] 

Tags: PINN, Symbolic Regression

"This paper introduces Universal PINNs (UPINN), a hybrid PINN-UDE framework for discovering unknown differential operators from sparse data using symbolic regression, achieving robust and accurate identification of hidden physics."

"本文提出 Universal PINNs (UPINN)，结合 PINN 和 UDE 进行符号微分算子发现，在稀疏数据下通过符号回归识别未知物理规律，具有强噪声鲁棒性和高精度。"

**9. Variational Autoencoding Neural Operators**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23841)\] \[[CODE]()\] 

Tags: Neural Operator, Super-Resolution, Unsupervised, Zero-Shot

"This paper introduces Variational Autoencoding Neural Operators (VANO), a discretization-agnostic framework that enables unsupervised functional data modeling with operator learning architectures, achieving state-of-the-art performance in zero-shot super-resolution of physical processes."

"本文提出了 Variational Autoencoding Neural Operators (VANO)，一种离散化无关的无监督学习框架，将变分自编码与神经算子结合，在物理过程的零样本超分辨率任务中实现了最先进的性能。"

**10. D-CIPHER: Discovery of Closed-form Partial Differential Equations**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/70676)\] \[[CODE]()\] 

Tags: Symbolic Regression

"D-CIPHER enables the discovery of closed-form high-order ODEs and PDEs from data without derivative estimation, offering a more robust and generalizable approach to equation discovery."

"D-CIPHER 通过无导数估计的优化策略，从数据中发现封闭形式的高阶 ODE 和 PDE，为方程发现提供更广泛适用的方法。"

**11. Self-Supervised Learning with Lie Symmetries for Partial Differential Equations**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/71235)\] \[[CODE]()\] 

Tags: Foundation Model, Lie Algebra, SSL

"This work introduces a self-supervised learning framework leveraging Lie symmetries to learn general-purpose PDE representations from heterogeneous data, improving parameter regression and time-stepping tasks."

"本文提出了一种基于 Lie 群对称性的自监督学习方法，用于从异构 PDE 数据中学习通用表示，以提升参数回归和时间步进任务的泛化能力。"

**12. Learning Differential Operators for Interpretable Time Series Modeling**: \[[KDD2022](https://dl.acm.org/doi/10.1145/3534678.3539245)\] \[[CODE]()\] 

Tags: Temporal

""

""

**13. Discovering Nonlinear PDEs from Scarce Data with Physics-encoded Learning**: \[[ICLR2022](https://iclr.cc/virtual/2022/poster/6855)\] \[[CODE]()\] 

Tags: Sparse Regression

"This paper proposes a physics-encoded discrete learning framework that combines deep convolutional-recurrent networks and sparse regression to robustly discover nonlinear PDEs from scarce and noisy data."

"本文提出了一种物理编码的离散学习框架，结合深度卷积-递归网络和稀疏回归，从稀缺和噪声数据中稳健地发现非线性 PDEs。"

**14. AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_33a854e247155d590883b93bca53848a.html)\] \[[CODE]()\] 

Tags: 

""

""

**15. Deep Energy-based Modeling of Discrete-Time Physics**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_98b418276d571e623651fc1d471c7811.html)\] \[[CODE]()\] 

Tags: 

""

""

**16. PDE-Net: Learning PDEs from Data**: \[[ICML2018](https://proceedings.mlr.press/v80/long18a.html)\] \[[CODE]()\] 

Tags: 

""

""

## 4. Analysis

**1. Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators**: \[[ICLR2025](https://openreview.net/forum?id=J9FgrqOOni)\] \[[CODE]()\] 

Tags: Error, Neural Operator

"This paper identifies discretization mismatch errors in neural operators and proposes CROP to enhance cross-resolution PDE learning."

"本文揭示了神经算子中的离散化失配误差，并提出 CROP 以提升跨分辨率 PDE 学习的可靠性。"

**2. On the expressiveness and spectral bias of KANs**: \[[ICLR2025](https://openreview.net/forum?id=ydlDRUuGm9)\] \[[CODE]()\] 

Tags: Kolmogorov-Arnold, Spectral Bias

"KANs exhibit stronger expressiveness and reduced spectral bias compared to MLPs, making them promising for PDE-related tasks requiring high-frequency accuracy."

"KAN 相较于 MLP 具有更强的表达能力和更小的谱偏差，使其在需要高频精度的 PDE 任务中更具潜力。"

**3. Quantitative Approximation for Neural Operators in Nonlinear Parabolic Equations**: \[[ICLR2025](https://openreview.net/forum?id=yUefexs79U)\] \[[CODE]()\] 

Tags: Approximation Theory, Neural Operator, Picard Iteration

"This work establishes a quantitative approximation theorem for neural operators in solving nonlinear parabolic PDEs, revealing their connection to Picard’s iteration and avoiding exponential model complexity growth."

"本文建立了神经算子在求解非线性抛物型 PDE 中的定量逼近定理，揭示其与 Picard 迭代的联系，并避免了指数级模型复杂度增长。"

**4. Adversarial Adaptive Sampling: Unify PINN and Optimal Transport for the Approximation of PDEs**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/19361)\] \[[CODE]()\] 

Tags: Optimal Transport, PINN, Sampling

""

""

**5. An operator preconditioning perspective on training in physics-informed machine learning**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18473)\] \[[CODE]()\] 

Tags: Precondition, Training

""

""

**6. Guaranteed Approximation Bounds for Mixed-Precision Neural Operators**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18680)\] \[[CODE]()\] 

Tags: Memory, Neural Operator, Precision

""

""

**7. Scaling physics-informed hard constraints with mixture-of-experts**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/17590)\] \[[CODE]()\] 

Tags: Foundation Model, Hard Constraints, PINN

""

""

**8. CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations**: \[[ICML2024](https://icml.cc/virtual/2024/poster/33364)\] \[[CODE]()\] 

Tags: Low-Rank Adaptation, Model Reduction

"CoLoRA’s continuous low-rank adaptation drastically accelerates PDE solution modeling while maintaining high accuracy, even with limited training data."

""

**9. Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs**: \[[ICML2024](https://proceedings.mlr.press/v235/mouli24a.html)\] \[[CODE]()\] 

Tags: Conservation, Neural Operator, OOD

"By encouraging diverse model predictions and incorporating physical constraints, DiverseNO and Operator-ProbConserv enhance out-of-domain PDE performance while providing reliable uncertainty estimates."

""

**10. Boosting Generalization in Parametric PDE Neural Solvers through Adaptive Conditioning**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95866)\] \[[CODE](https://geps-project.github.io/)\] 

Tags: Meta-Learning

"GEPS introduces an adaptive conditioning mechanism for parametric PDE solvers, significantly improving generalization across varying conditions through meta-learning and low-rank adaptation."

"GEPS 提出了一种自适应条件机制，通过元学习和低秩自适应优化提升参数化 PDE 求解器的泛化能力。"

**11. Can neural operators always be continuously discretized?**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/94369)\] \[[CODE]()\] 

Tags: Neural Operator

"This work establishes fundamental limitations on the continuous discretization of neural operators in Hilbert spaces and introduces strongly monotone neural operators as a solution to ensure discretization invariance."

"该工作揭示了希尔伯特空间中神经算子的离散化极限，并提出强单调神经算子作为保证离散化不变性的解决方案。"

**12. How does PDE order affect the convergence of PINNs?**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/96377)\] \[[CODE]()\] 

Tags: Convergence Guarantee, PINN

"This paper analyzes the negative impact of high PDE order on PINN convergence and proposes variable splitting as a strategy to improve training stability by reducing differential order."

"该研究分析了高阶 PDE 对 PINN 收敛性的负面影响，并提出变量拆分方法，通过降低微分阶数来提高训练稳定性。"

**13. The Challenges of the Nonlinear Regime for Physics-Informed Neural Networks**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95966)\] \[[CODE]()\] 

Tags: NTK, Optimization, PINN

"This paper reveals fundamental differences in the NTK behavior of PINNs for linear and nonlinear PDEs, demonstrating the necessity of second-order optimization methods for improved convergence."

"该研究揭示了 PINN 在线性和非线性 PDEs 上的 NTK 行为差异，并证明了使用二阶优化方法提升收敛性的必要性。"

**14. Understanding the Expressivity and Trainability of Fourier Neural Operator: A Mean-Field Perspective**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/95243)\] \[[CODE]()\] 

Tags: Neural Operator

"This paper provides a mean-field analysis of the Fourier Neural Operator, revealing its expressivity and trainability through an ordered-chaos phase transition, offering practical insights for stable training."

"本文通过均场理论分析 FNO 的表达能力和可训练性，揭示其有序-混沌相变特性，并为稳定训练提供实践指导。"

**15. Improved Training of Physics-Informed Neural Networks Using Energy-Based Priors: a Study on Electrical Impedance Tomography**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/10758)\] \[[CODE](https://rooshenasgroup.github.io/eit_ebprior/)\] 

Tags: Inverse, PINN

"This paper introduces an energy-based prior to stabilize and accelerate PINN training for solving the ill-posed inverse problem in Electrical Impedance Tomography (EIT), achieving a 10x speedup in convergence."

"本文提出了一种基于能量模型（EBM）的先验方法，用于 稳定和加速 PINN 在 电阻抗层析成像（EIT） 逆问题上的训练，使其收敛速度提高 10倍 并显著提升求解精度。"

**16. Nonlinear Reconstruction for Operator Learning of PDEs with Discontinuities**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/12026)\] \[[CODE]()\] 

Tags: Neural Operator

"This paper establishes theoretical lower bounds for operator learning of PDEs with discontinuities and demonstrates that nonlinear reconstruction methods, such as FNO and Shift-DeepONet, significantly improve approximation efficiency."

"本文建立了针对间断 PDEs 的算子学习的理论下界，并证明了非线性重构方法（如 FNO 和 Shift-DeepONet）能显著提高逼近效率。"

**17. Gradient Descent Finds the Global Optima of Two-Layer Physics-Informed Neural Networks**: \[[ICML2023](https://icml.cc/virtual/2023/poster/25183)\] \[[CODE]()\] 

Tags: Convergence Guarantee, Optimization, PINN

"This paper provides a rigorous convergence analysis for gradient descent in two-layer PINNs, proving that it finds the global optima under over-parameterization for various PDEs."

"本文对两层 PINNs 的梯度下降收敛性进行了严格分析，证明在过参数化情况下，对多种 PDE 任务均能找到全局最优解。"

**18. Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23759)\] \[[CODE]()\] 

Tags: PINN, Sampling

"This paper introduces the "Propagation Hypothesis" to explain PINN failures and proposes the R3 adaptive sampling strategy to mitigate propagation failures, significantly improving accuracy and stability."

"本文提出“传播假设”来解释 PINNs 失效模式，并引入 R3 自适应采样策略，以缓解传播失败问题，在多个 PDE 任务上显著提高求解精度与稳定性。"

**19. MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23480)\] \[[CODE]()\] 

Tags: Multi-Scale, Optimization, PINN

"This paper introduces MultiAdam, a scale-invariant optimizer that balances PDE and boundary loss in PINNs, significantly improving accuracy and convergence stability."

"本文提出 MultiAdam 这一尺度不变优化器，在 PINN 训练中平衡 PDE 损失和边界损失，大幅提升求解精度与收敛稳定性。"

**20. Neural Network Approximations of PDEs Beyond Linearity: A Representational Perspective**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24628)\] \[[CODE]()\] 

Tags: PINN

"This work extends neural network approximation theory to nonlinear PDEs, proving that two-layer networks can efficiently approximate solutions while avoiding the curse of dimensionality."

"本研究扩展了神经网络近似理论至非线性 PDE，证明了两层网络能够高效逼近解，并避免维度灾难。"

**21. Globally injective and bijective neural operators**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72751)\] \[[CODE]()\] 

Tags: Inverse, Neural Operator

"This work analyzes the injectivity and surjectivity of neural operators in function spaces, proving their invertibility and offering theoretical insights for applications in uncertainty quantification and inverse problems."

"本文研究了神经算子在函数空间中的单射和满射性质，并证明了其可逆性，为神经算子在不确定性量化和反问题求解中的应用提供了理论保障。"

**22. Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/70991)\] \[[CODE]()\] 

Tags: Meta-Learning, PINN

"This work introduces a hypernetwork-based meta-learning approach to improve the training efficiency of low-rank PINNs and enhance their generalization in parameterized PDE solving, particularly addressing the "failure modes" of PINNs."

"本文提出了一种基于超网络的元学习方法，以优化低秩 PINNs 的训练效率，并提升其在参数化 PDE 求解中的泛化能力，特别是解决 PINNs 训练中的“失败模式”问题。"

**23. Training neural operators to preserve invariant measures of chaotic attractors**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72621)\] \[[CODE](https://github.com/roxie62/neural_operators_for_chaos)\] 

Tags: Neural Operator

"This work introduces a novel approach to training neural operators that preserves the statistical invariance of chaotic attractors, leading to more stable and physically consistent long-term predictions."

"本文提出了一种新方法，使神经算子能够保持混沌吸引子的统计不变性，从而提高长期预测的稳定性和物理一致性。"

**24. Machine Learning For Elliptic PDEs: Fast Rate Generalization Bound, Neural Scaling Law and Minimax Optimality**: \[[ICLR2022](https://iclr.cc/virtual/2022/poster/6547)\] \[[CODE]()\] 

Tags: Generalization Bound

"This paper establishes fast rate generalization bounds and minimax optimality for deep learning methods, including PINNs and a modified Deep Ritz Method, in solving elliptic PDEs, demonstrating improved scaling laws and statistical limits."

"本文通过引入快速率泛化界和极小极大最优性，分析了深度学习方法（包括 PINNs 和改进的 Deep Ritz Method）在求解椭圆型 PDEs 中的表现，展示了改进的缩放规律和统计极限。"

**25. Generic bounds on the approximation error for physics-informed (and) operator learning**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/54367)\] \[[CODE]()\] 

Tags: Error, Neural Operator, PINN

"This paper introduces a general framework that provides the first rigorous approximation error bounds for PINNs, DeepONets, and FNOs, demonstrating their capability to overcome the curse of dimensionality when approximating nonlinear parabolic PDEs like the Allen-Cahn equation."

"本研究提出了一个通用框架，首次为 PINNs、DeepONets 和 FNOs 等物理约束和算子学习方法提供了严格的误差界，证明了它们在逼近非线性抛物型 PDEs（如 Allen-Cahn 方程）时可以克服维数灾难。"

**26. Unravelling the Performance of Physics-informed Graph Neural Networks for Dynamical Systems**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/55670)\] \[[CODE]()\] 

Tags: GNN, PINN, Zero-Shot

"This paper systematically evaluates 13 physics-informed graph neural networks (GNNs) for dynamical systems, revealing that explicit constraints and decoupling of kinetic and potential energies significantly enhance model generalization and performance."

"本文系统评估了13种物理信息图神经网络 (Physics-Informed GNNs) 在动态系统中的表现，揭示了显式约束和能量解耦对模型泛化和性能的显著提升。"

**27. Characterizing possible failure modes in physics-informed neural networks**: \[[NeurIPS2021](https://neurips.cc/virtual/2021/poster/26497)\] \[[CODE]()\] 

Tags: PINN

""

""

**28. On the Representation of Solutions to Elliptic PDEs in Barron Spaces**: \[[NeurIPS2021](https://neurips.cc/virtual/2021/poster/26803)\] \[[CODE]()\] 

Tags: 

""

""

**29. Parametric Complexity Bounds for Approximating PDEs with Neural Networks**: \[[NeurIPS2021](https://neurips.cc/virtual/2021/poster/26558)\] \[[CODE]()\] 

Tags: Bound

""

""

## 5. Data & Benchmarks

**1. Active Learning for Neural PDE Solvers**: \[[ICLR2025](https://openreview.net/forum?id=x4ZmQaumRg)\] \[[CODE]()\] 

Tags: Active Learning, Efficiency

"AL4PDE provides a structured benchmark for active learning in neural PDE solvers, demonstrating improved data efficiency, reduced errors, and reusable training datasets."

"AL4PDE 构建了神经 PDE 求解中的主动学习基准，显著提升数据效率、降低误差，并生成可复用的数据集。"

**2. Open-CK: A Large Multi-Physics Fields Coupling benchmarks in Combustion Kinetics**: \[[ICLR2025](https://openreview.net/forum?id=A23C57icJt)\] \[[CODE]()\] 

Tags: Multi-Physics

"Open-CK provides a high-resolution multi-physics benchmark for AI-driven PDE solving in combustion kinetics, enabling advancements in turbulence modeling and fire prediction."

"Open-CK 构建了一个高分辨率多物理场基准数据集，推动 AI 在燃烧动力学 PDE 求解中的应用，促进湍流建模与火灾预测研究。"

**3. Accelerating Data Generation for Neural Operators via Krylov Subspace Recycling**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/18516)\] \[[CODE]()\] 

Tags: Data Generation, Krylov, Neural Operator

""

""

**4. Accelerating PDE Data Generation via Differential Operator Action in Solution Space**: \[[ICML2024](https://proceedings.mlr.press/v235/dong24d.html)\] \[[CODE]()\] 

Tags: Data Generation, Differential Operator, Efficiency, Neural Operator

"DiffOAS leverages a small number of base solutions and their differential operator actions to rapidly generate precise PDE datasets, dramatically cutting down computational overhead."

""

**5. APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/97550)\] \[[CODE](https://github.com/tum-pbs/apebench)\] 

Tags: Autoregressive

"APEBench provides a comprehensive benchmark suite for autoregressive PDE emulators, integrating differentiable simulations and emphasizing rollout performance analysis for evaluating long-term temporal generalization."

"APEBench 提供了一个全面的基准测试套件，用于评估自回归 PDE 预测模型，集成可微模拟，并强调滚动误差分析，以研究长时间预测的泛化能力。"

**6. ChaosBench: A Multi-Channel, Physics-Based Benchmark for Subseasonal-to-Seasonal Climate Prediction**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/97481)\] \[[CODE](https://leap-stc.github.io/ChaosBench)\] 

Tags: Climate

"ChaosBench provides a physics-based benchmark for evaluating the subseasonal-to-seasonal (S2S) predictability of data-driven climate emulators, emphasizing physical consistency and long-term forecasting challenges."

"ChaosBench 提供了一个基于物理的基准测试，用于评估数据驱动气候模拟器在次季节至季节 (S2S) 预测中的可预测性，强调物理一致性和长期预测挑战。"

**7. PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/97621)\] \[[CODE](https://github.com/i207M/PINNacle)\] 

Tags: Multi-Scale, PINN

"PINNacle establishes the largest benchmark for systematically evaluating physics-informed neural networks (PINNs) on a diverse set of PDEs, providing insights into their strengths, weaknesses, and future research directions."

"PINNacle 构建了迄今为止最大规模的 PINN 基准测试，系统评估其在多种 PDE 任务上的表现，揭示其优势、劣势及未来研究方向。"

**8. The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/97882)\] \[[CODE](https://github.com/PolymathicAI/the_well)\] 

Tags: Large-Scale

"The Well provides a large-scale, diverse collection of physics simulation datasets, enabling systematic benchmarking and evaluation of machine learning models for PDE-based physical systems."

"The Well 提供了大规模、多样化的物理模拟数据集，支持基于 PDE 物理系统的机器学习模型的系统化基准测试与评估。"

**9. General Covariance Data Augmentation for Neural PDE Solvers**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23727)\] \[[CODE](https://github.com/VLSF/augmentation)\] 

Tags: Data Augmentation, Efficiency, Training

"This work proposes a general covariance-based data augmentation strategy for neural PDE solvers, reducing reliance on expensive PDE solvers and improving generalization."

"本研究提出了一种基于广义协变性的神经 PDE 求解器数据增强方法，降低了对昂贵 PDE 求解器的依赖，并提升了泛化能力。"

**10. BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/73722)\] \[[CODE](https://github.com/HPCForge/BubbleML)\] 

Tags: Benchmark

"This work introduces BubbleML, a high-fidelity multiphase multiphysics simulation dataset, along with benchmarks for optical flow analysis and neural PDE solvers, providing a valuable resource for ML applications in phase change phenomena."

"本文提出了一个高保真度的多相多物理模拟数据集 BubbleML，并提供了光流分析和神经 PDE 求解两个基准任务，为机器学习在相变现象中的应用提供了重要资源。"

**11. ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/73569)\] \[[CODE](https://leap-stc.github.io/ClimSim)\] 

Tags: Climate, Hybrid, Multi-Scale

"本文提出 ClimSim，这是目前最大规模的多尺度气候模拟数据集，专为机器学习与物理混合模拟（hybrid ML-physics simulation）设计，有助于提升气候模拟器的精度和长期预测能力。"

"ClimSim is the largest multi-scale climate simulation dataset designed for hybrid ML-physics research, facilitating high-fidelity emulation of atmospheric processes and improving climate model projections."

**12. Lie Point Symmetry Data Augmentation for Neural PDE Solvers**: \[[ICML2022](https://icml.cc/virtual/2022/poster/17313)\] \[[CODE]()\] 

Tags: Data Augmentation, Lie Algebra

"This paper introduces Lie Point Symmetry Data Augmentation (LPSDA) to enhance the sample efficiency of neural PDE solvers by leveraging mathematically grounded symmetry transformations, significantly reducing the need for costly high-quality ground truth data."

"本文提出了一种基于李点对称 (Lie Point Symmetry) 的数据增强方法 (LPSDA)，通过数学上严格的对称变换显著提高神经 PDE 求解器的数据样本效率，减少对高质量真值数据的需求。"

**13. PDEBench: An Extensive Benchmark for Scientific Machine Learning**: \[[NeurIPS2022](https://neurips.cc/virtual/2022/poster/55731)\] \[[CODE](https://github.com/pdebench/PDEBench)\] 

Tags: Benchmark, Inverse

"PDEBench offers a comprehensive benchmark suite covering a wide range of time-dependent and independent PDE problems, enabling fair comparison and extensibility for scientific machine learning (Scientific ML) methods."

"PDEBench 提供了一个全面的基准测试套件，涵盖多种时间依赖和独立的偏微分方程 (PDE) 问题，支持科学机器学习 (Scientific ML) 方法的公平对比与扩展。"

## 6. Applications

**1. ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs**: \[[ICLR2024](https://iclr.cc/virtual/2024/poster/17438)\] \[[CODE]()\] 

Tags: Advection, Climate

""

""

**2. Generalizing Weather Forecast to Fine-grained Temporal Scales via Physics-AI Hybrid Modeling**: \[[NeurIPS2024](https://nips.cc/virtual/2024/poster/93990)\] \[[CODE](https://github.com/black-yt/WeatherGFT)\] 

Tags: Climate, Hybrid, Temporal

"WeatherGFT combines PDE-based physical evolution with AI correction, enabling weather forecasting models to generalize across finer-grained temporal scales beyond their training data."

"WeatherGFT 结合基于 PDE 的物理演化与 AI 误差修正，使天气预报模型能够泛化到超出训练数据范围的更精细时间尺度。"

**3. Accurate medium-range global weather forecasting with 3D neural networks**: \[[Nature2023](https://www.nature.com/articles/s41586-023-06185-3)\] \[[CODE]()\] 

Tags: Climate

""

""

**4. Koopman Neural Operator Forecaster for Time-series with Temporal Distributional Shifts**: \[[ICLR2023](https://iclr.cc/virtual/2023/poster/11432)\] \[[CODE](https://github.com/google-research/google-research/tree/master/KNF)\] 

Tags: Neural Operator, Temporal

"This paper introduces Koopman Neural Forecaster (KNF), a deep sequence model leveraging Koopman theory to enhance robustness against temporal distributional shifts in time-series forecasting."

"本文提出了一种基于 Koopman 理论 的 神经预测器（KNF），通过学习全局与局部算子，提高时间序列预测对 时变分布偏移 的鲁棒性。"

**5. ClimaX: A foundation model for weather and climate**: \[[ICML2023](https://icml.cc/virtual/2023/poster/24136)\] \[[CODE](https://github.com/microsoft/ClimaX)\] 

Tags: Climate, Foundation Model, Multi-Scale, Transformer

"This work introduces ClimaX, a foundation model for weather and climate science that leverages heterogeneous datasets and a Transformer-based architecture to improve generalization across multiple forecasting tasks."

"本研究提出了 ClimaX，一个面向天气和气候科学的基础模型，利用异构数据和 Transformer 结构，在多种预测任务上展现了强大的泛化能力。"

**6. Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere**: \[[ICML2023](https://icml.cc/virtual/2023/poster/23618)\] \[[CODE](https://github.com/NVIDIA/torch-harmonics)\] 

Tags: Neural Operator, Temporal

"This paper introduces Spherical Fourier Neural Operators (SFNOs), an equivariant and grid-invariant extension of FNOs to spherical geometries, which achieves stable and physically plausible long-term forecasting of atmospheric dynamics."

"本文提出了 Spherical Fourier Neural Operators (SFNOs)，一种针对球面几何的等变且网格无关的 FNO 扩展，在大气动力学的长期预测中实现了稳定且物理上合理的动态建模。"

**7. DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting**: \[[NeurIPS2023](https://nips.cc/virtual/2023/poster/71410)\] \[[CODE](https://github.com/Rose-STL-Lab/dyffusion)\] 

Tags: Diffusion, Temporal

"DYffusion introduces a dynamics-informed diffusion model that enhances probabilistic spatiotemporal forecasting by leveraging temporal structure, improving accuracy and efficiency in complex physical systems."

"本文提出 DYffusion，一种动力学增强的扩散模型，利用时间动态信息提高概率性时空预测的准确性和计算效率，适用于复杂物理系统的长期滚动预测。"

**8. NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/70820)\] \[[CODE](https://github.com/vLAR-group/NVFi)\] 

Tags: 3D, Neural Fields

"NVFi introduces neural velocity fields to model dynamic 3D scenes from multi-view videos, leveraging physics-informed constraints to enable future frame extrapolation, semantic decomposition, and motion transfer."

"本文提出 NVFi，一种神经速度场学习方法，通过多视角视频学习 3D 物理动态场景，结合物理约束进行优化，能够支持未来帧预测、3D 语义分解和运动转移。"

**9. ResoNet: Noise-Trained Physics-Informed MRI Off-Resonance Correction**: \[[NeurIPS2023](https://neurips.cc/virtual/2023/poster/72112)\] \[[CODE](https://github.com/mikgroup/ResoNet)\] 

Tags: Medical Imaging

"ResoNet introduces a physics-informed deep learning framework for MRI off-resonance correction, leveraging synthetic data for training and enabling robust generalization across anatomies and contrasts."

"ResoNet 提出了一种物理驱动的深度学习方法来校正 MRI 中的离共振伪影，利用合成数据进行训练，实现对不同解剖结构和对比度的泛化能力。"

**10. Combining Differentiable PDE Solvers and Graph Neural Networks for Fluid Flow Prediction**: \[[ICML2020](https://icml.cc/virtual/2020/poster/6802)\] \[[CODE]()\] 

Tags: Fluid, GNN

""

""

**11. Learning to Simulate Complex Physics with Graph Networks**: \[[ICML2020](https://icml.cc/virtual/2020/poster/6849)\] \[[CODE]()\] 

Tags: GNN

""

""

**12. Lorentz Group Equivariant Neural Network for Particle Physics**: \[[ICML2020](https://arxiv.org/abs/2006.04780)\] \[[CODE]()\] 

Tags: 

""

""

**13. JAX MD: A Framework for Differentiable Physics**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_83d3d4b6c9579515e1679aca8cbc8033.html)\] \[[CODE]()\] 

Tags: 

""

""

**14. Learning Composable Energy Surrogates for PDE Order Reduction**: \[[NeurIPS2020](https://nips.cc/virtual/2020/public/poster_0332d694daab22e0e0eaf7a5e88433f9.html)\] \[[CODE]()\] 

Tags: 

""

""
