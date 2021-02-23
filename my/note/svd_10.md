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