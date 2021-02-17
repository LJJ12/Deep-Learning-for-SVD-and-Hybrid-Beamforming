#### 个人总结，个人向！！！，帮助自己复现论文使用

版本：keras 2.2.5 相应的源码

https://keras-zh.readthedocs.io/

1. #### keras.models

   https://keras.io/zh/models/about-keras-models/

2. #### keras.optimizers

   提供给网络需要的参数（学习率，衰减，momentum，梯度下降等）

   

3. #### Input

   Input():用来实例化一个keras张量

   ```python
   def Input(shape=None, batch_shape=None,
             name=None, dtype=None, sparse=False,
             tensor=None):
   ```

   shape: 形状元组（整型），不包括batch size。for instance, shape=(32,) 表示了预期的输入将是一批32维的向量。

   batch_shape: 形状元组（整型），包括了batch size。for instance, batch_shape=(10,32)表示了预期的输入将是10个32维向量的批次。

   name: 对于该层是可选的名字字符串。在一个模型中是独一无二的（同一个名字不能复用2次）。如果name没有被特指将会自动生成。

   dtype: 预期的输入数据类型

   sparse: 特定的布尔值，占位符是否为sparse

   tensor: 可选的存在的向量包装到Input层，如果设置了，该层将不会创建一个占位张量。

   实例：

   ```python
   # this is a logistic regression in Keras
   # Returns  A tensor.
   x = Input(shape=(32,))
   y = Dense(16, activation='softmax')(x)
   model = Model(x, y)
   ```

   

4. #### Lamda

   使用keras来搭建模型主要两种方法

   1. 较为简单的序列模型Sequential（该方法适用于搭建简单的模型https://blog.csdn.net/mogoweb/article/details/82152174）
   2. 使用Keras函数式的API（该方法最为常用）

     但是上述两种都只能使用keras中layer的各种实现子类层，例如Dense,Conv2D等等。但在实际的论文中，存在自定义的层,这时候必须想办法将其转换成keras中的Layer. 

   一般有两种方法:

   1. 直接定义类class然后继承Layer
   2. 直接使用Lambda函数。这里主要讲使用Lambda函数的方法，该方法比较简单

   参看链接https://blog.csdn.net/qq_37541097/article/details/102469546

   其中自己使用的是单独定义方法，然后再使用lamda转换

   ```
   def myFunc(parameters):
   	# 具体函数操作
   	
   X_out = Lambda(myFunc)(X_in)
   ```

   

   

5. #### Dense 

   全连接层 http://keras-cn.readthedocs.io/en/latest/layers/core_layer/

   https://keras-zh.readthedocs.io/layers/core/

   Just your regular densely-connected NN layer.

   ```python
   def __init__(self, units, 节点数
                    activation=None, 激活函数
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    **kwargs):
   ```

   - **units**: 正整数，输出空间维度。
   - **activation**: 激活函数 (详见 [activations](https://keras-zh.readthedocs.io/activations/))。 若不指定，则不使用激活函数 (即，线性激活: `a(x) = x`)。
   - **use_bias**: 布尔值，该层是否使用偏置向量。
   - **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
   - **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
   - **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
   - **bias_regularizer**: 运用到偏置向量的的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
   - **activity_regularizer**: 运用到层的输出的正则化函数 (它的 "activation")。 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
   - **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
   - **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。

   

6. #### concatenate

   连接两个数组，axis=n表示**从第n个维度进行拼接**

   ```python
   X_in = concatenate([X_temp, T_RF, T_BB, R_RF, R_BB], axis=1)
   ```

   https://blog.csdn.net/leviopku/article/details/82380710

   

7. #### Conv2D

   2D convolution layer (e.g. spatial convolution over images). 

   2D 卷积层（例如，图像上的空间卷积）

   ```python
   def __init__(self, filters, 卷积核的数目
                    kernel_size, 卷积核的宽度和长度
                    strides=(1, 1),默认卷积的步长
                    padding='valid',补0策略，“valid”代表只进行有效的卷积，即对边界数据不处理。
                    data_format=None,图像的通道维的位置
                    dilation_rate=(1, 1),指定dilated convolution中的膨胀比例
                    activation=None,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    **kwargs):
   ```

   实例

   ```python
   X_input = Input(input_shape)
   X = ZeroPadding2D((7, 7))(X_input)
   # 卷积核数目   卷积核长宽   步长   命名
   X = Conv2D(32, (3, 3),
              strides=(1, 1), name='conv0')(X)
   ```
