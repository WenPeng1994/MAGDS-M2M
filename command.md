## 无连接远程运行
nohup python3 main_v1.py >> ./logging/main_v1_output.log 2>&1 &

## 查看运行情况
lsof -i:63720

## 关闭运行的进程
kill -9 进程号

## 激活base
source ~/.bashrc

## 在base中激活其他环境
conda activate Evans0

## 查看CPU使用情况
top

## 查看GPU使用情况
nvidia-smi

## 查看无连接远程运行
jobs -l

## 查看文件倒数指定行
tail -n 100 ./logging/main_v17_2024-04-29_23\:06.log

## 查看文件的前几行
head -n 100 ./logging/main_v39_output.log

## 查看整个文件
cat ./logging/main_v39_output.log