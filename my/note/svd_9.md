#### train_on_batch

```
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

运行一批样品的单次梯度更新。

__参数_

- **x**: 训练数据的 Numpy 数组（如果模型只有一个输入）， 或者是 Numpy 数组的列表（如果模型有多个输入）。 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
- **y**: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
- **sample_weight**: 可选数组，与 x 长度相同，包含应用到模型损失函数的每个样本的权重。 如果是时域数据，你可以传递一个尺寸为 (samples, sequence_length) 的 2D 数组， 为每一个样本的每一个时间步应用不同的权重。 在这种情况下，你应该在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- **class_weight**: 可选的字典，用来映射类索引（整数）到权重（浮点）值，以在训练时对模型的损失函数加权。 这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。

**返回**

标量训练误差（如果模型只有一个输入且没有评估标准）， 或者标量的列表（如果模型有多个输出 和/或 评估标准）。 属性 `model.metrics_names` 将提供标量输出的显示标签。



### test_on_batch

```
test_on_batch(x, y, sample_weight=None)
```

在一批样本上测试模型。

**参数**

- **x**: 测试数据的 Numpy 数组（如果模型只有一个输入）， 或者是 Numpy 数组的列表（如果模型有多个输入）。 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
- **y**: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
- **sample_weight**: 可选数组，与 x 长度相同，包含应用到模型损失函数的每个样本的权重。 如果是时域数据，你可以传递一个尺寸为 (samples, sequence_length) 的 2D 数组， 为每一个样本的每一个时间步应用不同的权重。

**返回**

标量测试误差（如果模型只有一个输入且没有评估标准）， 或者标量的列表（如果模型有多个输出 和/或 评估标准）。 属性 `model.metrics_names` 将提供标量输出的显示标签。

#### tf.zeros_like()

```
keras.backend.zeros_like(x, dtype=None, name=None)
```

实例化与另一个张量相同尺寸的全零变量。

**参数**

- **x**: Keras 变量或 Keras 张量。
- **dtype**: 字符串，返回的 Keras 变量的类型。 如果为 None，则使用 x 的类型。
- **name**: 字符串，所创建的变量的名称。

**返回**

一个 Keras 变量，其形状为 x，用零填充。

```
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

#### ones_like

```
keras.backend.ones_like(x, dtype=None, name=None)
```

实例化与另一个张量相同形状的全一变量。

**参数**

- **x**: Keras 变量或张量。
- **dtype**: 字符串，返回的 Keras 变量的类型。 如果为 None，则使用 x 的类型。
- **name**: 字符串，所创建的变量的名称。

**返回**

一个 Keras 变量，其形状为 x，用一填充。

**例子**

```
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```

#### identity

```
keras.backend.identity(x, name=None)
```

返回与输入张量相同内容的张量。

**参数**

- **x**: 输入张量。
- **name**: 字符串，所创建的变量的名称。

**返回**

一个相同尺寸、类型和内容的张量。

#### temp_alpha = sess.run(temp_alpha, feed_dict={alpha: 0.05})

```
import tensorflow as tf
a = tf.add(2, 5)
b = tf.multiply(a, 3)
with tf.Session() as sess: 
    c = sess.run(b)
print(c)

replace_dict = {a:15}
with tf.Session() as sess:
    d = sess.run(b,feed_dict=replace_dict)
print(d)

```

```
sess = tf.Session()
alpha = tf.placeholder(tf.float32)
# 此时是一个张量 
temp_alpha = tf.zeros_like(alpha)
# 返回的结果是一个标量
temp_alpha = sess.run(temp_alpha, feed_dict={alpha: 0.05})
print(temp_alpha)
```

#### ELU

```
f(x) =  alpha * (exp(x) - 1.) for x < 0,
f(x) =  x                     for x >= 0
```

