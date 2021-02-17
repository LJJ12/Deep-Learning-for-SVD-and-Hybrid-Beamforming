左奇异矩阵可以用于行数的压缩。相对的，右奇异矩阵可以用于列数即特征维度的压缩

关于信道模型的知识

https://blog.csdn.net/weixin_43935696/article/details/109723978?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161244832216780255233471%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161244832216780255233471&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-1-109723978.pc_search_result_cache&utm_term=%E5%87%A0%E4%BD%95%E4%BF%A1%E9%81%93%E6%A8%A1%E5%9E%8B&spm=1018.2226.3001.4187

#### PRELIMINARIES: S V D AND HYBRID BF

##### Optimum BF Using Unconstrained SVD

1. Analog and digital beamformers of a hybrid BF system need to be designed based on the constraints of power and finite precision phase shifters, which are used in the RF domain

```
混合BF系统的模拟和数字波束形成需要在功率和有限精度移相器的约束下进行设计
```

2. By selecting the achieved rate as our metric, our goal is to design beamformers at the Tx and Rx (TRF, TBB, RRF, RBB), which maximize the rate defined in (5) while the following constraints are satisfied:

![image-20210204113704691](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204113704691.png)

- Due to the usage of phase shifters, the entries of TRF and RRFmust have constant modulus.![image-20210204113907837](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204113907837.png)

  

- Elements of each column in TRFand RRFare represented as quantized phase shifts, where each phase shifter is controlled by an Nq-bit input.

```
TRFand RRFare中每个列的元素表示为量化相移，其中每个相移器由nq位输入控制。
```

- n(m)th row of the RF precoding matrix at the Tx(Rx), which cor-
  responds to the phase shifts of the n(m)th antenna of the TRF(RRF), can be written as ![image-20210204114206311](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204114206311.png)for some ![image-20210204114246969](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204114246969.png)

- The  power  constraint  must  be  satisfied,![image-20210204114321173](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204114321173.png)

#### D. mmW Channel Model

1. Various studies [43], [44] have shown that mmW channels have limited scattering due to the high free-space path loss.In this model, each scatterer
   contributes a single propagation path between the Tx and the
   Rx.

```
各种研究[43]、[44]表明，毫米波通道由于高自由空间路径损耗而具有有限的散射。在这个模型中，每个散射体在Tx和Rx之间贡献一个单一的传播路径。
```

2. ![image-20210204115052859](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204115052859.png)

```
S is the number of scatterers 
ρ is the average path-loss between the Tx and the Rx
a communication system with NT and NR antennas at the Tx and Rx
gsis the complex gain of the s th path with Rayleigh distribution具有瑞利分布的sth路径的复增益
```

![image-20210204115513988](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204115513988.png)s =1,2,...,S.G denotes the average power gain.

![image-20210204115604881](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204115604881.png)

![image-20210204115614332](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204115614332.png)

 are the array response vectors at the Tx and the Rx.

φs∈ [0,2π] and θs∈ [0,2π] indicate the sth path’s azimuth Angle of Arrival (AoA) and Angle of Departure (AoD)

For more details of the geometric channel model we refer the reader to [45], [46].

paper 45 和 46 中的更多查看

### III. DL FOR SVD APPROXIMA TIONS

leverage DL to effectively estimate the best rank-k approximation of a matrix H

```
利用DL有效地估计矩阵H的最佳k秩近似
```

![image-20210204120642655](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204120642655.png)![image-20210204120652320](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204120652320.png)

只取前k个元素就用来表示H

#### A. DNN for Rank-k Matrix Approximation

1. 正交矩阵是方块矩阵，行向量和列向量皆为正交的单位向量,行向量皆为正交的单位向量，任意两行正交就是两行点乘结果为0，而因为是单位向量，所以任意行点乘自己结果为1。

#### B. Low-Complexity DNN for Rank-k Matrix Approximation

1. DNN-i is trained to estimate singular value σiand corresponding singular vectors ui and vi of a given matrix H

   训练DNN-i 来估计矩阵H的第i个奇异值和第i个左右奇异向量

2.  流程![image-20210204124047172](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204124047172.png)

| dnn-i | input | estimate    |
| :---: | ----- | ----------- |
|       |       |             |
|   1   | H     | σ'1,u'1,v'1 |
|   2   | H'2   | H'2=H-H'1   |
|   3   | H'3   | H'3=H-H'2   |

所有的H都是一样维度的矩阵，每次相减得到一个矩阵，这个矩阵包含的元素

![image-20210204182059302](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210204182059302.png)

### Experimental Study of DNNs for SVD

evaluate the performance of the proposed DNN architectures for the SVD

#### Data Generation

1.  8000 training and 2000 testing channel matrices

```
8000测试和2000个测试通道矩阵
几何信道模型
```

2. DL Model 图构建

```
输入节点数 都是2NrNt
输出节点数 
rank-k  k(2Nr+Nt+1) lc-rank-k 2(Nr+Nt+1)   rank-1 2(Nr+Nt+1)

图配置
多层convs+ 0.4 dropout layers + fc
convs和fc激活函数=》 elu
learning rate 0.0001
penalty rate 0.01
adaptive learning rate optimization adam
tf
```

3. Comparison of Training and T est Losses维度增加，过拟合之前迭代次数也可增加，不过预测是错误率也同时有增加

   