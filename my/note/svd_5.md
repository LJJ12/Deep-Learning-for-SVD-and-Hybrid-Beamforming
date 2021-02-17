### DL FOR HYBRID BF

we minimize the Frobenius dis-
tance between the rank-k approximations obtained with the
unconstrained and hybrid beamformers instead of maximizing
the rate directly.

```
我们将无约束和混合波束成型机得到的秩k近似之间的Frobenius距离最小化，而不是直接使速率最大化。
```

1. The DNN gets the channel matrix H =>Nr×Nt as the input

2.  transforms that into a real- valued matrix with Nr× 2Nt size

3. consists of multiple convolutional layers and one fully-connected layer

4. ```
   estimate the L largest singular values
   the unnormalized values of the BB precoder
   the unnormalized values of the BB combiner
   the unquantized values of the RF precoder
   the unquantized values of the RF combiner
   ```

![image-20210207153048570](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210207153048570.png)

#### Incorporation of RF Constraints

We propose four approaches to formulate the quantization as a differentiable function。denote the ith element of the unquantized and vectorized RF **precoder** estimated by the DNN as 
$$
\hat{x}_i^{RF} = c_re^{j\alpha_i}。where  \\
c_t = \frac 1 {\sqrt{N_t}} \quad is\quad the \quad modulus \quad \alpha_i 
$$
 is the phase of the ith element.在解码端也一样，查看论文即可

1. Quantization Approach 1

```
a combination of step and piece-wise linear functions 
阶跃线性函数和分段线性函数的组合

当γ趋于0时，量子化方法1开始表现为均匀量子化。当γ = 0，量化是在训练和测试阶段以完全相同的方式实现。γ越接近1，可微区越大,哪将允许更新与射频预编码器和组合器相关的权值在训练期间的每一次迭代.
```

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210207164548322.png)

2. we replace each step function in the uniform quantization with a sigmoid function in the second quantization approach

```
在第二种量化方法中，我们用sigmoid函数代替均匀量化中的每个阶函数.
```

3. Since the weights are not updated during forward propagation, we use step functions to apply uniform quantization in the third approach for forward propagation.During backpropagation, we use a linear combination of sigmoid functions

```
由于在前向传播过程中权重没有更新，我们使用阶跃函数在前向传播的第三种方法中应用均匀量化。
在反向传播过程中，我们使用sigmoid函数的线性组合
```

4. Finally, we propose a fourth quantization approach, which assigns αito one of 2Nq quantization points probabilistically during forward propagation.

```
最后，我们提出了一种第四量子化方法，该方法在正向传播过程中概率地将α分配到两个量子化点之一。
```

