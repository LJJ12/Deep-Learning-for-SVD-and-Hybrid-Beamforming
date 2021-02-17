python安装技巧

 pip install matplotlib -i https://pypi.doubanio.com/simple/

python -m pip install tensorflow -i https://pypi.douban.com/simple



 pip install     -i https://pypi.doubanio.com/simple/



```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
```



```
conda  create  -n xxx_name python=3.7 
conda  create  -n PILenv python=2.7 

在G:\anaconda\envs
```



```
conda install numpy pandas scikit-learn lightgbm matplotlib
```



```
conda create -n 文件夹 python=3.6（指定python的版本）
```



```
conda activate demo_pypro
或者使用 activate demo_pypro
```



```
pdb.set_trace()
c 
n
s step in 
r return
q
```



```
activate  tensorflow


// 以下都是在环境内进行的

pip  install  ipykernel

python -m ipykernel install --user --name tianchi_1 --display-name tianchi_1
python -m ipykernel install --user --name exercise --display-name exercise
```



```
查看有那些环境

conda remove -n xxx.name --all

退出环境
deactivate
```

```
python -m pip install tensorflow-gpu -i https://pypi.douban.com/simple

```

