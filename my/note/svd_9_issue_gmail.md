github

```
## 背景

- 描述你希望解决的问题的现状
- 附上相关的 issue 地址

## 思路

描述大概的解决思路，可以包含 API 设计和伪代码等

## 跟进

后续编辑，附上对应的 Pull Request 地址，可以用 `- [ ] some task` 的方式。
```

你好 ，我是来自中国西南石油大学-计算机科学学院-计算机科学专业-研一学生-邵永杰。

我目前工作是尝试使用深度学习来加速svd。关于你所著文章Deep Learning for SVD and Hybrid Beamforming非常感兴趣。但在阅读文献过程中，发现你文章所添加的源码地址已失效。

目前使用的你在GitHub上的源码，版本信息python=3.6,tensorflow=1.14,keras=2.2.5.在运行时候出现错误：TypeError: Value passed to parameter 'shape' has DataType float32 not in list of allowed values: int32, int64

查阅后我发现是个简单错误，只需要替换第182行，187行，192行，197行中"/"符号为"//"即可，确保了Dense层filter参数为int32或者int64。

如下：

```python
 T_RF = Dense(n_y-2*Num_Stream)(X)
 T_RF = ELU(alpha=1.0)(T_RF)
 T_RF = Dropout(0.1)(T_RF)
 # change "/" as "//",ensure the type of the result is integer   
 T_RF = Dense((n_y-2*Num_Stream)//4, activation='sigmoid')(T_RF)
    
 T_BB = Dense(n_y-2*Num_Stream)(X)
 T_BB = ELU(alpha=1.0)(T_BB)
 T_BB = Dropout(0.1)(T_BB)
# change "/" as "//",ensure the type of the result is integer  
 T_BB = Dense((n_y-2*Num_Stream)//2, activation='linear')(T_BB)
    
    
 R_RF = Dense(n_y-2*Num_Stream)(X)
 R_RF = ELU(alpha=1.0)(R_RF)
 R_RF = Dropout(0.1)(R_RF)
# change "/" as "//",ensure the type of the result is integer  
 R_RF = Dense((n_y-2*Num_Stream)//4, activation='sigmoid')(R_RF)
    
 R_BB = Dense(n_y-2*Num_Stream)(X)
 R_BB = ELU(alpha=1.0)(R_BB)
 R_BB = Dropout(0.1)(R_BB)
# change "/" as "//",ensure the type of the result is integer  
 R_BB = Dense((n_y-2*Num_Stream)//2, activation='linear')(R_BB)
```





最后，在此请问你是否原因分享其他DNN架构以及量化层的代码给我，邮箱:

202021000429@stu.swpu.edu.cn. 那将极大有助于我加速svd的求解。

​																									真诚感谢

Yongjie  Shao



gmail

```
Source code sharing request from Shao Yongjie, a student from China Southwest Petroleum University（https://www.swpu.edu.cn）

Although using gmail is not formal, this is the only way I can find  to contact you, Dr. Ture Peken.

I am Yongjie Shao from Southwest Petroleum University.Now I am a first-year postgraduate student of computer science.

My current job is to try to use deep learning to solve SVD. I am very interested in your article "Deep Learning for SVD and Hybrid Beamforming".But in the process of reading the literature, I found that the source code address added in your 

article is invalid: https://www.dropbox.com/sh/v0gs7ba0qq5 × 168/AACyqRoCz5m3fhpF-azkbn3Qa?dl=0.The creative ideas proposed in your article are very helpful for me to learn and improve my own knowledge and skills.

I have sent you a request under your githubrepository address before, including some simple understanding of your source code.But the code on github does not include other ideas in your article, it is a bit difficult for me to reproduce.Therefore, 

please do you intend to continue to share the source code of the other two DNN architectures and three quantization layers on dropbox. Or share it to my email: 202021000429@stu.swpu.edu.cn or syjeaapple@gmail.com That will greatly help 

my current study. thank you very much sincerely.


```

