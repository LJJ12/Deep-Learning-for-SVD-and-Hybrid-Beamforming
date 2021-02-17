Deep Learning for SVD and Hybrid Beamforming

个人学习总结，有错误理解，后续修改。

将无线信号（电磁波）只按特定方向传播的技术叫做波束成型(beamforming) https://zhuanlan.zhihu.com/p/21800662

波束成型和毫米波技术可谓是天作之合，使用毫米波可以给信号传输带来更大的带宽，波束成型则能解决频谱利用问题，使得5G通讯如虎添翼



波束合成算法的逻辑图

![img](https://img-blog.csdnimg.cn/20190504112112476.png)

波束成形实际上就是把原始的发送信号改变其幅度相位再进行叠加，从而得到想要的天线发送信号。而改变幅度和相位，数学上就可以简单的表示为与一个复数相乘。而叠加的过程，则可以简洁的使用矩阵乘法来表示https://blog.csdn.net/weixin_39274659/article/details/89810132

####   个人理解论文核心思想



```
The main idea of this work is to formulate the hybrid BF as a constrained SVD problem since the SVD based unconstrained BF constructs an upper bound on the maximum achievable rates
由于基于无约束BF的奇异值分解在最大可达率上构造了一个上界，因此本工作的主要思想是将混合BF表示为一个有约束的奇异值分解问题
we aim to study the potential of ML approaches for the SVD and hybrid BF.
我们的目的是研究ML方法求解SVD和混合BF的潜力。
```

1 使用SVD 给信道矩阵降维

2 但是求解SVD的传统过程计算量较大，使用DL 来加速这个过程

#### 三种不同的DNN架构

1. 输入矩阵通过单个DNN网络计算，直接输出所有的奇异值和奇异向量。
2. 整体包含K个DNN,每一个DNN都被训练来预测最大的奇异值和相关联的左右奇异向量。 低复杂度
3. 输入矩阵到单个DNN,循环迭代的输出一个奇异值和奇异向量。低复杂度+进一步简化SVD操作

#### K是含义

1. ##### 奇异值分解

   $$
   A = UΣV^T , (1) \\
   A =》 m*n阶,U =>m*m，Σ=>m*n,V=>n*n \\
   U^TU=I  V^TV=I的m阶与n阶酉矩阵\\
   {(Σ)_{ii}} = \delta_i,其他位置的元素均为0，\delta_i为非负且满足σ1⩾σ2⩾...⩾0 \\
   其中U的列向量u_i称为A的左奇异向量, V的列向量v_i称为A的右奇异向量, σi称为奇异值. \\
   矩阵的秩就等于非零奇异值的个数，但是奇异值σ跟特征值类似，在矩阵Σ中也是从大到小排列，\\
   而且σ的减少特别的快，在很多情况下，前10\%奇异值的和就占了全部的奇异值之和的99\%以上了。\\
   也就是说，我们也可以用前r大的奇异值来近似描述矩阵,这能减少存储消耗。在很小的损失下，提取矩阵特征。 
   $$

2. 使用低秩矩阵近似矩阵A
   $$
   在A = UΣV^T矩阵分解中,假设给定一个秩为r的矩阵A, 欲求其最优k秩近似矩阵A‘，k<= r(k可以远小于r)\\
   min_{A'∈R^{m×n}}∥A−A'∥_F  , (2)
   $$

   $$
   对A做奇异值分解后，将奇异矩阵Σ中的    (r    -     k ) 个最小的奇异值=0 .\\
   获的矩阵Σ_k,只保留K个最大的奇异值，则\\
   A_k=U_kΣ_kV^T_k ,(3) \\
   其中U_k和V_k分别是算式（1）前k列组成的矩阵，A_k就是A的最优k秩近似矩阵A'
   $$

3. 参考博客连接

   ```
   https://www.cnblogs.com/lhtan/p/7998662.html
   https://www.cnblogs.com/fionacai/p/5767973.html
   ```



#### 四种量化方法，避免量化之后，出现梯度消失的情况

4. We consider the case where finite-precision phase shifters areused in the RF domain, which restricts the analog beamformers to have constant modulus and quantized phase values. Therefore,quantization layers are included in the proposed DNN for hybrid BF.

   ```
   量化的意思：量化在数字信号处理领域，是指将[信号](https://baike.baidu.com/item/信号/32683)的连续取值（或者大量可能的离散取值）近似为有限多个（或较少的）离散值的过程。量化主要应用于从连续信号到[数字信号](https://baike.baidu.com/item/数字信号/915663)的转换中。[连续信号](https://baike.baidu.com/item/连续信号/6597074)经过采样成为[离散信号](https://baike.baidu.com/item/离散信号/6613954)，离散信号经过量化即成为数字信号。注意离散信号通常情况下并不需要经过量化的过程，但可能在值域上并不离散，还是需要经过量化的过程 。
   ```

   但是non-differentiability of the discretization operation（离散化运算的不可微），梯度消失To circumvent this issue, we propose four quantization approaches

1. 阶跃和分段线性函数

2. consider a soft quantization by using a combination of several sigmoid functions with different parameters during both forward as well as backward propagation

  ```
  考虑在正向和反向传播过程中使用几个具有不同参数的sigmoid函数的组合进行软量化
  ```

3. use step function in the forward propagation while incorporating sigmoid functions with different parameters during backward propagation

  ```
  在正向传播中使用阶跃函数，而在反向传播中使用带有不同参数的sigmoid函数
  ```

4. implement a stochastic quantization approach [37] during forward propagation while replacing with a straight-through estimator [38] during backpropagation.

  ```
  在正向传播过程中实现一个随机量化方法[37]，而在反向传播过程中用一个直接估计器[38]代替。
  ```

  

#### 结果分析指标

1. provide the time complexity analysis for the proposed DNN architectures for SVD and compare their time complexities with the conventional SVD algorithms


```
对所提出的DNN结构进行时间复杂度分析，并与传统的SVD算法进行时间复杂度比较  have a smaller time
complexity than the traditional SVD approaches while the number of transmit and receive antennas increases, and the other parameters remain constant.
```



