import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#  生成虚拟数据
import numpy as np
# 1000行 20列
x_train = np.random.random((10000, 20))
# [0，10) 整数
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)
x_test = np.random.random((1000, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#  设定优化器参数
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 配置编译器参数
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)


model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'