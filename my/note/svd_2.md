将无线信号（电磁波）只按特定方向传播的技术叫做波束成型(beamforming) https://zhuanlan.zhihu.com/p/21800662

波束成型和毫米波技术可谓是天作之合，使用毫米波可以给信号传输带来更大的带宽，波束成型则能解决频谱利用问题，使得5G通讯如虎添翼



波束合成算法的逻辑图

![波束合成图](https://img-blog.csdnimg.cn/20190504112112476.png)

波束成形实际上就是把原始的发送信号改变其幅度相位再进行叠加，从而得到想要的天线发送信号。而改变幅度和相位，数学上就可以简单的表示为与一个复数相乘。而叠加的过程，则可以简洁的使用矩阵乘法来表示https://blog.csdn.net/weixin_39274659/article/details/89810132



https://www.cnblogs.com/pinard/p/6251584.html

#### 结果分析指标

1. provide the time complexity analysis 

```
show
that the proposed DNN based approaches have a smaller time
complexity than the traditional SVD approaches while the
number of transmit and receive antennas increases, and the
other parameters remain constant
结果表明，在发射天线和接收天线数量增加，其他参数保持不变的情况下，基于DNN的方法比传统的SVD方法具有更小的时间复杂度
```

2. implement three DNN architectures for SVD using CNNs and discuss the impact of mini-batch size, the number of hidden layers,
   and training iterations size on accuracy

```
使用cnn为SVD实现三种DNN架构，并讨论小批量大小、隐藏层数量和训练迭代大小对精度的影响
```

3. With the geometric channel model, we simulate the proposed DNN based hybrid
   BF algorithm and compare its rates with the unconstrained BF, three conventional hybrid BF algorithms [9], [10], [39], an ML-aided hybrid BF algorithm based on CE optimiza-
   tion [25], two DL-based hybrid BF algorithms [27], [30]an autoencoder based hybrid BF algorithm [28].

   ```
   基于DNN的仿真的波束合成算法与其他8中算法 比较
   ```

   

4. compare the performance of the proposed DNN based SVD approaches with the traditional SVD algorithms in terms of the time complexity and memory requirements

```
比较了基于DNN的奇异值分解算法与传统奇异值分解算法在时间复杂度和内存需求方面的性能
```

#### Notion

1. e stands for Euler’s number and j denotes √−1. N?μ, σ2? and N (m,R) are a complex Gaussian random scalar with mean μ and variance σ2and a complex Gaussian random vector with mean m and covariance R, respectively. R and C denote the set of real and complex numbers, respectively.

```
e表示欧拉数，j表示√−1。N ?μ、σ2 ?和N (m,R)分别是均值μ和方差σ2的复高斯随机标量和均值m和协方差R的复高斯随机向量。R和C分别表示实数和复数的集合。
```

##### SVD

1. **r ≤ l =min{NT,NR}**{此处就是SVD的应用特性，只需要前10%的奇异值的和就可以近似表示原矩阵的特征}，there exists (i) a unitary matrix U ∈ CNR×NR; (ii) a diagonal matrix Σ ∈ CNR×NTwith non-negative numbers on its diagonal; (iii) a unitary matrix V ∈ CNT×NT![image-20210202153439580](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210202153439580.png)

此处的j不是上文中的Ｊ = -1的开根号

##### SVD性质

1. 对于奇异值,它跟我们特征分解中的特征值类似，在奇异值矩阵中也是按照从大到小排列，而且奇异值的减少特别的快，在很多情况下，前10%甚至1%的奇异值的和就占了全部的奇异值之和的99%以上的比例。也就是说，我们也可以用最大的k个的奇异值和对应的左右奇异向量来近似描述矩阵
   $$
   A_{m*n} = U_{m*m}Σ_{m*n}V^T_{n*n} \approx U_{m*k}Σ_{k*k}V^T_{k*n}  \\
   k 一般远小于 m 和 n
   $$
   ![image-20210202201653721](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210202201653721.png)

   https://www.cnblogs.com/pinard/p/6251584.html

##### Optimum BF Using Unconstrained SVD

1. Source codes for the experiments are available at: https://www.dropbox.
   com/sh/v0gs7ba0qq5 × 168/AACyqRoCz5m3fhpF-azkbn3Qa?dl=0

   ```swift
   curl -L -o newName.zip https://www.dropbox.com/sh/[folderLink]?dl=1
   
   curl -L -o newName.zip https://www.dropbox.com/sh/v0gs7ba0qq5 × 168/AACyqRoCz5m3fhpF-azkbn3Qa?dl=1
   ```

   下载教程https://www.jianshu.com/p/50bccca07d50

   

2. For the scope of this paper, we focus on the achieved rate, which is maximized by selecting the singular vectors of H as the beamformers of this system.In particular, the optimum beamformers are found by maximizing the rate R![image-20210202162114111](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210202162114111.png)

   Topt= VLand Ropt= ULdenote the optimum unconstrained precoder and combiner of this system. Here,
   VL∈ CNT×Land UL∈ CNR×Lare L most signifi- cant right and left singular vectors of H [42], respectively.
   Cn= R∗optRoptis the post-processing noise covariance matrix.

   The average total transmit power is denoted as P,s  s ∈ CL×1is the vector of transmitted symbols and satisfies ![image-20210202162459380](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210202162459380.png)

```
在本文的范围内，我们关注的是实现速率，通过选择H的奇异向量作为系统的波束形成器，实现速率是最大的.特别地，通过使速率R最大化来寻找最佳波束形成器.
Topt= VLand Ropt= uld表示该系统的最优无约束预编码器和组合器。这里，VL∈CNT×Land UL∈CNR×Lare L H[42]最重要的左右奇异向量。Cn= R∗optRoptis后处理噪声协方差矩阵
平均总发射功率记为P
```

##### Hybrid BF Using Constrained SVD

1. A Tx with NTantennas and LTRF chains com-municates with a Rx with NRantennas and LRRF chains.
   We assume there are L data streams such that L ≤ LT≤ NT and L ≤ LR≤ NR

```
一个带有NTantennas和LTRF链的Tx与一个带有NRantennas和LRRF链的Rx通信。我们假设有L≤LT≤NT和L≤LR≤NR的数据流
```

2. We denote the mmW channel between the Tx and Rx with H ∈ CNR×NT

```
我们用H∈CNR×NT表示Tx与Rx之间的毫米波通道
```

3. ith BF vector of the RF precoder are given as [TRF]:,i, i = 1,...,LT

```
RF预编码器的第ith BF向量
```

