## keras

#### Conv2D

1. filters ：整数，输出空间的维度 （即卷积中滤波器的输出数量）

2. kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值

3. strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长

   

#### MaxPooling2D

1. pool_size: 整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。

#### 切片

1. b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象。a = [0,1,2,3,4,5,6,7,8,9] ,b = a[1:3] 那么，b的内容是 [1,2]。

   ```
   当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
   当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
   当i,j都缺省时，a[:]就相当于完整复制一份a了
   ```

2. b = a[i:j:s]这种格式，i,j与上面的一样，但s表示步进，缺省为1.所以a[i:j:1]相当于a[i:j]

   ```
   当s<0时，
   i缺省时，默认为-1. 
   j缺省时，默认为-len(a)-1
   所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序序列
   ```

   



## tf

#### tf.where

```
tf.where(
    condition,
    x=None,
    y=None,
    name=None
)
```

根据condition返回x或y中的元素.

如果x和y都为None,则该操作将返回condition中true元素的坐标.坐标以二维张量返回,其中第一维(行)表示真实元素的数量,第二维(列)表示真实元素的坐标.请记住,输出张量的形状可以根据输入中的真实值的多少而变化.索引以行优先顺序输出.

如果两者都不是None,则x和y必须具有相同的形状.如果x和y是标量,则condition张量必须是标量.如果x和y是更高级别的矢量,则condition必须是大小与x的第一维度相匹配的矢量,或者必须具有与x相同的形状.

condition张量作为一个可以选择的掩码(mask),它根据每个元素的值来判断输出中的相应元素/行是否应从 x (如果为 true) 或 y (如果为 false)中选择.

如果condition是向量,则x和y是更高级别的矩阵,那么它选择从x和y复制哪个行(外部维度).如果condition与x和y具有相同的形状,那么它将选择从x和y复制哪个元素.

函数参数：

- condition：一个bool类型的张量(Tensor).
- x：可能与condition具有相同形状的张量；如果condition的秩是1,则x可能有更高的排名,但其第一维度必须匹配condition的大小.
- y：与x具有相同的形状和类型的张量.
- name：操作的名称(可选).

返回值：

如果它们不是None,则返回与x,y具有相同类型与形状的张量；张量具有形状(num_true, dim_size(condition)).

可能引发的异常：

- ValueError：当一个x或y正好不是None.

#### tf.floor

```
floor(
    x,
    name=None
)
```

返回不大于 x 的元素最大整数.

参数：

- x：张量,必须是下列类型之一：half、float32、float64.
- name：操作的名称(可选).

返回：

​			该函数返回一个张量,与 x 具有相同的类型.

#### tf.stack

```
tf.stack(
    values,
    axis=0,
    name='stack'
)
```

将秩为 R 的张量列表堆叠成一个秩为 (R+1) 的张量.

将 values 中的张量列表打包成一个张量,该张量比 values 中的每个张量都高一个秩,通过沿 axis 维度打包.给定一个形状为(A, B, C)的张量的长度 N 的列表；

如果 axis == 0,那么 output 张量将具有形状(N, A, B, C).如果 axis == 1,那么 output 张量将具有形状(A, N, B, C).

例如：

```
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```

这与 unpack 相反,numpy 相当于：

```
tf.stack([x, y, z]) = np.stack([x, y, z])
```

函数参数：

- values：具有相同形状和类型的 Tensor 对象列表.
- axis：一个 int,要一起堆叠的轴,默认为第一维,负值环绕,所以有效范围是[-(R+1), R+1).
- name：此操作的名称(可选).

函数返回值：

- output：与 values 具有相同的类型的堆叠的 Tensor.

可能引发的异常：

- ValueError：如果 axis 超出范围 [ - (R + 1),R + 1),则引发此异常.

#### tf.complex

```
complex( 
    real, 
    imag, 
    name=None
 )
```

将两个实数转换为复数.

给定 real 表示复数的实部的张量和 imag 表示复数的虚部的张量,该操作的返回形式为 \(a + bj \)的元数字的复数,其中 a 表示 real 部分,b 表示 imag 部分.

输入的张量 real 和 imag 必须具有相同的形状.

例如：

```
# 张量 'real' 是 [2.25, 3.25]
# 张量 `imag` 是 [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```

#### ARGS：

- real：张量.必须是以下类型之一：float32,float64.
- imag：张量.必须与 real 具有相同的类型.
- name：操作的名称(可选).

#### 返回：

返回 complex64 或 complex128 类型的张量.

## tf.concat

```
concat ( 
    values , 
    axis , 
    name = 'concat' 
)
```

将张量沿一个维度串联.

将张量值的列表与维度轴串联在一起.如果 values[i].shape = [D0, D1, ... Daxis(i),Dn],则连接结果有形状.

```
[D0, D1, ... Raxis, ...Dn]
```

当

```
Raxis = sum(Daxis(i))
```

也就是说,输入张量的数据将沿轴维度连接.
输入张量的维数必须匹配, 并且除坐标轴外的所有维度必须相等.

例如：

```
T1 =  [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] ] 
T2 =  [ [ 7 , 8 , 9 ] , [ 10 , 11 , 12 ] ] 
tf.concat([T1 ,T2] ,0) == >  [[1 , 2 ,3 ],[4 ,5 ,6],[7 ,8 ,9],[10 ,11,12]] 
tf.concat([T1 ,T2] ,1) == >  [[ 1 ,2 ,3 ,7 ,8 ,9 ],[4 ,5 ,6,10 ,11 ,12]]

＃张量 t3 的形状[2,3] 
＃张量 t4 的形状[2,3] 
tf.shape(tf.concat([ t3 , t4 ] , 0 )) == >  [ 4 , 3 ] 
tf.shape( tf.concat([t3 ,t4 ] , 1 )) == >  [ 2 , 6 ]
```

注意：如果沿着新轴连接,请考虑使用堆栈:

```
tf.concat ([ tf.expand_dims (t ,axis) for t in tensors] ,axis)
```

可以重写为

```
tf.stack(tensors,axis = axis)
```

ARGS：

- values：张量对象或单个张量列表.
- axis：0 维 int32 张量,要连接的维度.
- name：操作的名称(可选).

返回：

由输入张量的连接引起的张量.

#### tf.expand_dims

用于增加维度

维度增加一维，可以使用`tf.expand_dims(input, dim, name=None)`函数

```
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

#### tf.matmul

```
matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
)
```

将矩阵 a 乘以矩阵 b,生成a * b

输入必须在任何转换之后是 rank> = 2 的张量,其中内部 2 维度指定有效的矩阵乘法参数,并且任何其他外部维度匹配.

两个矩阵必须是相同类型.支持的类型有：float16,float32,float64,int32,complex64,complex128.

通过将相应的标志之一设置为 True,矩阵可以被转置或 adjointed(共轭和转置).默认情况下,这些都是 False.

如果一个或两个矩阵包含很多的零,则可以通过将相应的 a_is_sparse 或 b_is_sparse 标志设置为 True 来使用更有效的乘法算法,默认为 false.这个优化仅适用于具有数据类型为bfloat16 或 float32 的纯矩阵(rank 为2的张量).

例如：

```
# 2-D tensor `a`
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                      [4. 5. 6.]]
# 2-D tensor `b`
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                         [9. 10.]
                                                         [11. 12.]]
c = tf.matmul(a, b) => [[58 64]
                        [139 154]]

# 3-D tensor `a`
a = tf.constant(np.arange(1, 13, dtype=np.int32),
                shape=[2, 2, 3])                  => [[[ 1.  2.  3.]
                                                       [ 4.  5.  6.]],
                                                      [[ 7.  8.  9.]
                                                       [10. 11. 12.]]]

# 3-D tensor `b`
b = tf.constant(np.arange(13, 25, dtype=np.int32),
                shape=[2, 3, 2])                   => [[[13. 14.]
                                                        [15. 16.]
                                                        [17. 18.]],
                                                       [[19. 20.]
                                                        [21. 22.]
                                                        [23. 24.]]]
c = tf.matmul(a, b) => [[[ 94 100]
                         [229 244]],
                        [[508 532]
                         [697 730]]]

# Since python >= 3.5 the @ operator is supported (see PEP 465).
# In TensorFlow, it simply calls the `tf.matmul()` function, so the
# following lines are equivalent:
d = a @ b @ [[10.], [11.]]
d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
```

参数：

- a：类型为 float16,float32,float64,int32,complex64,complex128 和 rank > 1的张量.
- b：与 a 具有相同类型和 rank.
- transpose_a：如果 True,a 在乘法之前转置.
- transpose_b：如果 True,b 在乘法之前转置.
- adjoint_a：如果 True,a 在乘法之前共轭和转置.
- adjoint_b：如果 True,b 在乘法之前共轭和转置.
- a_is_sparse：如果 True,a 被视为稀疏矩阵.
- b_is_sparse：如果 True,b 被视为稀疏矩阵.
- name：操作名称(可选).

返回：

该函数返回与 a 和 b 具有相同类型的张量,其中每个最内矩阵是 a 和 b 中对应矩阵的乘积,例如,如果所有转置或伴随的属性为 False：

```
output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j
```

Note：这是矩阵乘积,而不是元素的乘积.

可能引发的异常：

- ValueError：如果 transpose_a 和 adjoint_a,或者 transpose_b 和 adjoint_b 都设置为 True.

## Keras