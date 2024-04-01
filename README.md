# FBSNNJ: Solving PIDE with DeepLearning

This is a partial integral differential equation solver based on deep learning. 


## Introduction
Consider the following second-order semilinear parabolic partial differential equation.

$$\left\{
\begin{aligned}
-\partial_tu-\mathcal{L}u-f(\cdot,\cdot,u,\sigma^\top\nabla_x u) & =0,    & (t,x) & \in[0,T]\times\mathbb{R}^d \\
u(T,x)                                                           & =g(x), & x     & \in\mathbb{R}^d
\end{aligned}
\right.$$


Let $d\geq1$ be the dimension, $T>0$ be the terminal condition, $g:\mathbb{R}^d\mapsto\mathbb{R}$, and $f:[0,T]\times\mathbb{R}^d\times\mathbb{R}\times\mathbb{R}^d\times\mathbb{R}\mapsto\mathbb{R}$. The differential operator $\mathcal{L}$ describes the evolution of the function $u$ with respect to time $t$ and space $x$, where the first term is the diffusion term, the second term is the convection term, and the third term is the integral term, defined specifically as follows,

$$\begin{aligned}
    \mathcal{L}u := & \; \frac{1}{2}\mathrm{Tr}(\sigma\sigma^\top\nabla_x^2u)+\langle b,\nabla_xu \rangle          \\
                    & + \int_E (u(t,x+\beta(t,x,e)) -u(t,x) - \langle \nabla_x u, \beta(t,x,e)\rangle )\lambda(de)
\end{aligned}$$

where $\mathrm{Tr}(\cdot)$ denotes the trace of a matrix, $\sigma:[0,T]\times\mathbb{R}^d\mapsto\mathbb{M}^d$ is a matrix function, $\nabla_x^2u$ is the Hessian matrix of $u$, $\langle \cdot,\cdot \rangle$ denotes the inner product, and $b(t,x):[0,T]\times\mathbb{R}^d\mapsto \mathbb{R}^d$ is a vector function.

For the integral term, define the random variable space $E \triangleq \mathbb{R}^l\setminus\{0\}$, $\beta(t,x,e) :[0,T]\times\mathbb{R}^d\times E\mapsto \mathbb{R}^d$ is a vector function. Let $\mathcal{E}$ be the Borel field corresponding to $E$, and let the $\sigma$-finite measure $\lambda(de)$ on $(E,\mathcal{E})$ satisfy:
$$\int_E (1\wedge |e|^2) \lambda(de) <\infty,$$
where $|\cdot|$ denotes the $L^2$ norm.

## Methodology

First, we utilize the nonlinear Feynman-Kac formula to transform PIDEs into forward-backward stochastic differential equations with jumps (FBSDEJs), and use the Euler scheme to obtain the discrete format of FBSDEJs. Subsequently, we transform the problem of solving the discrete FBSDEJs into an optimization problem, and use deep learning techniques to solve the optimization problem, ultimately achieving numerical solutions for PIDEs.


Compared with existing research, the main innovation of this paper lies in the use of a smaller network and the handling of the integral term. For the integral term, we consider first expanding the integrand using Taylor series before integration, using the gradient term to approximate the non-local integral, and simplifying the integral calculation. The numerical solution method in this paper only uses one network to approximate the solution of PIDEs, with the differential term solved using automatic differentiation techniques, and the integral term calculated based on the differential term. Since only one network is used, compared to numerical methods that use separate neural networks to fit the differential and integral terms, the total parameter size used in the numerical method of this paper is smaller, which is more conducive to neural network parameter optimization. Numerical experiments show that the forward-backward deep neural network format can obtain numerical solutions with a relative error on the order of $10^{-3}$.

## Dependencies
```bash
pip install -r requirements.txt
```

- matplotlib==3.8.0
- munch==4.0.0
- numpy==1.26.4
- pandas==2.2.1
- scipy==1.12.0
- tensorflow==2.14.0


## Files
- [SolverNN.py](SolverNN.py): Solver based on forward-backward stochastic differential equation with jump.
- [Network.py](Network.py): Feed forward neural network.
- [Tools.py](Tools.py): Useful tools.
- Other: Specific PIDE equations and solving code.

## Usage
### For the solver file that has been written
Run code and get the result. 

### For the new equation

1. Create new file and import necessary packages.
    ```python
    import numpy as np
    import tensorflow as tf
    import logging
    import scipy.stats as ss

    import Tools 
    from Network import *
    from SolverNN import FBSNNJ
    ```
2. Custom equation solving classes
    - Inherits the FBSNNJ class
    - `def __init__`: Custom parameter
    - Custom functions in PIDE 
        - `x_init_generator`: initial points
        - `Element_jump`: Element jump strength
        - `Forward_jump`: Forward jump integral calculation
        - `jump_bate_fun`: Jump function in backward SDE
        - `Forward_SDE`: Discrete form of forward SDE
        - `Backward_SDE`: Discrete form of backward SDE
        - `Terminal_condition`: Terminal condition
        - `Solution`: Ground truth solution
        - `model_approximate`: Neural network approximation
3. Instantiate the class and train


## Relative works
- Jiequn Han, Arnulf Jentzen, and Weinan E: [Solving high-dimensional partial differential equations using deep learning](https://doi.org/10.1073/pnas.1718942115)
- Maziar Raissi: [Forward–Backward Stochastic Neural Networks: Deep Learning of High-Dimensional Partial Differential Equations](https://doi.org/10.1142/9789811280306_0018)
- Liwei Lu, Hailong Guo, Xu Yang, Yi Zhu: [Temporal Difference Learning for High-Dimensional PIDEs with Jumps](https://arxiv.org/abs/2307.02766)