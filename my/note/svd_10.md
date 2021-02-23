#### tf.norm

```
norm(
    tensor,
    ord='euclidean',
    axis=None,
    keep_dims=False,
    name=None
)
```

计算向量、矩阵和张量的范数.

这个函数可以计算几个不同的向量范数(1-norm,Euclidean 或 2-norm,inf-norm,p> 0 的 p-norm)和矩阵范数(Frobenius,1-norm 和 inf -norm).

```
tf.norm(tf.matmul(R_RF_tmp, R_BB_tmp), axis=[1, 2]), axis=1)
```

参数：

- tensor：float32,float64,complex64,complex128 类型的张量.
- ord：范数的顺序.支持的值是“fro”、“euclidean”、0、1 、2、np.inf 和任意正实数,得到相应的 p-norm.缺省值是 'euclidean',如果张量是一个矩阵,则相当于 Frobenius 范数；如果是向量,则相当于 2-norm.一些限制适用：1、所述的 Frobenius 范数不是为向量所定义；2、若轴为 2 元组(矩阵范数),仅支持 “euclidean”、“fro”、1 、np.inf .有关如何计算在张量中存储的一批向量或矩阵的准则,请参见轴的说明.
- axis：如果 axis 是 None(默认值),那么输入被认为是一个向量,并且在张量的整个值集合上计算单个向量范数,即 norm(tensor,ord=ord)是等价于norm(reshape(tensor, [-1]), ord=ord).如果 axis 是 Python 整数,则输入被认为是一组向量,轴在张量中确定轴,以计算向量的范数.如果 axis 是一个2元组的 Python 整数,则它被认为是一组矩阵和轴,它确定了张量中的坐标轴,以计算矩阵范数.支持负数索引.示例：如果您在运行时传递可以是矩阵或一组矩阵的张量,则通过 axis=[-2,-1],而不是 axis=None 确保计算矩阵范数.
- keep_dims：如果为 True,则 axis 中指定的轴将保持为大小 1.否则,坐标轴中的尺寸将从 "输出" 形状中移除.
- name：操作的名字.

返回值：

- output：与张量具有相同类型的 Tensor,包含向量或矩阵的范数.如果 keep_dims 是 True,那么输出的排名等于张量的排名.否则, 如果轴为 none,则输出为标量；如果轴为整数,则输出的秩小于张量的秩；如果轴为2元组,则输出的秩比张量的秩低两倍.

可能引发的异常：

- ValueError：如果 ord 或者 axis 是无效的.

numpy 兼容性

大致相当于 numpy.linalg.norm.不支持：ord <= 0,矩阵的 2-norm,nuclear norm.

其他区别：1、如果轴为 None, 则将扁平的张量视为向量,而不考虑秩.2、明确支持 "euclidean" 范数作为默认值,包括高阶张量

#### tf.expand_dims

```
import tensorflow as tf
x =  tf.constant([[[1,2],[3,4]],
                 [[5,6],[7,8]]])
with tf.Session():
    print(x.shape)
    print(x.eval())
    
y = tf.expand_dims(x,1)
with tf.Session():
    print(y.shape)
    print(y.eval()) 
    
z = tf.expand_dims(x,0)
with tf.Session():
    print(z.shape)
```









#### tf.transpose

```
tf.transpose(
    a,
    perm=None,
    name='transpose',
    conjugate=False
)
```

置换 a,根据 perm 重新排列尺寸.

返回的张量的维度 i 将对应于输入维度 perm[i].如果 perm 没有给出,它被设置为(n-1 ... 0),其中 n 是输入张量的秩.因此,默认情况下,此操作在二维输入张量上执行常规矩阵转置.如果共轭为 True,并且 a.dtype 是 complex64 或 complex128,那么 a 的值是共轭转置和.

例如：

```
x = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(x)  # [[1, 4]
                 #  [2, 5]
                 #  [3, 6]]

# Equivalently
tf.transpose(x, perm=[1, 0])  # [[1, 4]
                              #  [2, 5]
                              #  [3, 6]]

# If x is complex, setting conjugate=True gives the conjugate transpose
x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                 [4 + 4j, 5 + 5j, 6 + 6j]])
tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                 #  [2 - 2j, 5 - 5j],
                                 #  [3 - 3j, 6 - 6j]]

# 'perm' is more useful for n-dimensional tensors, for n > 2
x = tf.constant([[[ 1,  2,  3],
                  [ 4,  5,  6]],
                 [[ 7,  8,  9],
                  [10, 11, 12]]])

# Take the transpose of the matrices in dimension-0
# (this common operation has a shorthand `matrix_transpose`)
tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
                                 #   [2,  5],
                                 #   [3,  6]],
                                 #  [[7, 10],
                                 #   [8, 11],
                                 #   [9, 12]]]
```

函数参数：

- a：一个 Tensor.
- perm：a 的维数的排列.
- name：操作的名称(可选).
- conjugate：可选 bool,将其设置为 True 在数学上等同于 tf.conj(tf.transpose(input)).

返回：

tf.transpose 函数返回一个转置 Tensor.