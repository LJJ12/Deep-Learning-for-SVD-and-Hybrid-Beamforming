import numpy as np
test=np.load('../../../Dataset/C_44_test.npy', encoding ="latin1")  #加载文件
doc = open('read_npy.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)