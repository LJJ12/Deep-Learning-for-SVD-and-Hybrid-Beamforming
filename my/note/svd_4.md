1. 什么是白噪声？
 答：白噪声是指功率谱密度在整个频域内均匀分布的噪声。白噪声或白杂讯，是一种功率频谱密度为常数的随机信号或随机过程。换句话说，此信号在各个频段上的功率是一样的，由于白光是由各种频率（颜色）的单色光混合而成，因而此信号的这种具有平坦功率谱的性质被称作是“白色的”，此信号也因此被称作白噪声。相对的，其他不具有这一性质的噪声信号被称为有色噪声。
理想的白噪声具有无限带宽，因而其能量是无限大，这在现实世界是不可能存在的。实际上，我们常常将有限带宽的平整讯号视为白噪音，因为这让我们在数学分析上更加方便。然而，白噪声在数学处理上比较方便，因此它是系统分析的有力工具。一般，只要一个噪声过程所具有的频谱宽度远远大于它所作用系统的带宽，并且在该带宽中其频谱密度基本上可以作为常数来考虑，就可以把它作为白噪声来处理。例如，热噪声和散弹噪声在很宽的频率范围内具有均匀的功率谱密度，通常可以认为它们是白噪声。
     高斯白噪声的概念——."白"指功率谱恒定;高斯指幅度取各种值时的概率p (x)是高斯函数
     高斯噪声——n维分布都服从高斯分布的噪声
     高斯分布——也称正态分布，又称常态分布。对于随机变量X，记为N（μ，σ2），分别为高斯分布的期望和方差。当有确定值时，p
 (x)也就确定了，特别当μ=0，σ2=1时，X的分布为标准正态分布。 

#### Comparison of Dropout and Max Pooling

1. rank-k test losses decrease slower.the smallest test losses are obtained with dropout when the DNN for rank-k approximation is trained more than 5000 iterations
2. he low-complexity DNN for rank-k  the smallest test losses are
   obtained when the dropout rate is 0.2.
3. the low-complexity DNN requires less generalization due to
   its simplicity compared to the DNN for rank-k approximation.
   Therefore, a lower dropout rate is required to achieve the best
   performance for the low-complexity DNN.

```
LC-Rank-k DNN 与秩k近似的DNN相比，复杂度较低的DNN简单，需要的泛化较少。因此，低复杂度的DNN需要更低的dropout rate才能获得最佳性能。，
```

#### Impact of Selected k Value During the Training and Testing

1. to estimate a smaller or a larger number of singular values and singular vectors than the selected k value during the training.the value
   of k in training must be at least equal to the value of k used for
   the testing.

```
在训练过程中估计比所选的k值更小或更多的奇异值和奇异向量 估计个数>k
训练中的k值必须至少等于用于测试的k值。
```

2. the DNN estimates a smaller number of singular values and singular vectors than the number of singular values.the DNN does not need to be retrained to estimate a smaller number of singular values and vectors than k used during the training. This result also shows that the test losses of the DNN based SVD approach do not change with the rank of the matrix when the size of the matrix remains the same

```
估计个数<k
DNN不需要再训练来估计比训练时使用的k更少的奇异值和向量。这一结果也表明，在矩阵大小不变的情况下，基于DNN的SVD方法的测试损耗不随矩阵的秩而变化
```

#### Performance of the Proposed Approach in Noisy Case

1. 30db的时候没有明显下降，当间隔为-10 ~ 30 ==》40分贝的时候，出现过拟合
2. compare the case when the SNR of AWGN is higher for the training matrices compared to the SNR of AWGN added to the test matrices.This result occurs since the overfitting increases more if the training data is less noisy compared to the test data.

```
比较训练矩阵的AWGN信噪比高于测试矩阵的AWGN信噪比的情况。
产生这种结果的原因是，如果训练数据的噪声比测试数据小，则过拟合增加更多。
```

####  Comparison of the Proposed Architectures for SVD

1. Comparison of the Proposed Training Approaches for the
   Low-Complexity DNN for Rank-k Approximation

#### Impact of Different Number of Convolutional Layers on the Accuracy of DNNs 

1. Impact of Different Number of Convolutional Layers on the Accuracy of DNNs

####  Impact of Different Sizes of Mini-Batches on the Accu-racy of DNNs

1.  In this section, we evaluate the performance
   of proposed DNNs in terms of accuracy with the different
   sizes of mini-batches

#### Comparison of Conventional Methods and DNN Based Approaches for SVD

1. Time Complexity Comparison

$$
[19]O(N^{2}_rN_t) [20]O(N_rN^{2}_t)
$$

2. time  complexity  of  the  training  and  test phases of a CNN with n convolutional layers are given  as 对于一个有n个卷积层的CNN，训练和测试阶段的时间复杂度为
   $$
   O(N*b*_{i=1}^n\sum_{i=1}^nm_{i-1}*f_i^2*m_i*l_i^2) \\
   O(\sum_{i=1}^nm_{i-1}*f_i^2*m_i*l_i^2)
   $$
   

