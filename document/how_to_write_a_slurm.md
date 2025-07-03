可以使用sinfo查看分区信息，当你输入sinfo指令之后，你可以看到类似：
(py310v2) yjiao@storage-hdd:~/SAE_Training/SAE_Training$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu          up 7-00:00:00      1   idle cpu
A100         up 7-00:00:00      1    mix node1
RTX3090      up 7-00:00:00      1 drain* node3
RTX3090      up 7-00:00:00      1    mix node2
RTX4090      up 7-00:00:00      4    mix node[4-7]
debug*       up 1-00:00:00      1 drain* node3
debug*       up 1-00:00:00      1    mix node2
ADA6000      up 7-00:00:00      2    mix node[8-9]
L40S         up 7-00:00:00      1    mix node11
L40S         up 7-00:00:00      1   idle node10
的信息

| 指令 | 含义 |
| :--- | :--- |
| `#!/bin/bash` | 指定脚本使用Bash shell来执行。 |
| `#SBATCH --job-name=...` | 给你的任务起一个名字，方便用`squeue`等命令查看。 |
| `#SBATCH --partition=RTX3090` | **非常重要**：指定你要提交任务到的分区（Partition/Queue）。这个名字（`RTX3090`）是**特定于你的集群的**，可能需要根据你所在集群的配置修改。 |
| `#SBATCH --nodes=1` | 请求使用1个计算节点。 |
| `#SBATCH --ntasks-per-node=1` | 在这个节点上只运行1个任务（进程）。对于单GPU训练，这通常是正确的设置。 |
| `#SBATCH --cpus-per-task=4` | 为你这1个任务分配4个CPU核心。这对于数据加载（DataLoader）很有帮助，可以防止CPU成为瓶颈。 |
| `#SBATCH --gres=gpu:1` | **关键指令**：请求通用资源（Generic Resource），这里是1块GPU。 |
| `#SBATCH --mem=16G` | 请求16GB的内存（RAM）。 |
| `#SBATCH --output=..._%j.out` | 指定标准输出（`print`语句等）写入的文件。`%j`是一个通配符，会被替换为任务的ID号。 |
| `#SBATCH --error=..._%j.err` | 指定标准错误（报错信息）写入的文件。`%j`同样会被替换为任务ID。 |
| `#SBATCH --time=4:00:00` | 设置任务的最长运行时间为4小时。如果超过这个时间，任务会被系统强制终止。 |
