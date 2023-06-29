# 训练与测试

- [训练与测试](#训练与测试)
  - [训练](#训练)
    - [使用单个 GPU 进行训练](#使用单个-gpu-进行训练)
    - [使用多个 GPU 进行训练](#使用多个-gpu-进行训练)
    - [使用多台机器进行训练](#使用多台机器进行训练)
      - [同一网络中的多台机器](#同一网络中的多台机器)
      - [使用 slurm 管理的多台机器](#使用-slurm-管理的多台机器)
  - [测试](#测试)
    - [使用单个 GPU 进行测试](#使用单个-gpu-进行测试)
    - [使用多个 GPU 进行测试](#使用多个-gpu-进行测试)
    - [使用多台机器进行测试](#使用多台机器进行测试)
      - [同一网络中的多台机器](#同一网络中的多台机器-1)
      - [使用 slurm 管理的多台机器](#使用-slurm-管理的多台机器-1)

## 训练

### 使用单个 GPU 进行训练

您可以使用 `tools/train.py` 在一台带有 CPU 和 GPU(可选) 的单机上训练模型。

下面是脚本的完整用法：

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

````{note}
默认情况下，MMAction2 更倾向于使用 GPU 而不是 CPU 进行训练。如果您想在 CPU 上训练模型，请清空 `CUDA_VISIBLE_DEVICES` 或将其设置为 -1 以使 GPU 对程序不可见。

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [ARGS]
```
````

| 参数                                  | 描述                                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件的路径。                                                                                                                                                    |
| `--work-dir WORK_DIR`                 | 保存日志和权重的目标文件夹。默认为与配置文件相同名称的文件夹，位于 `./work_dirs` 下。                                                                               |
| `--resume [RESUME]`                   | 恢复训练。如果指定了路径，则从该路径恢复，如果未指定，则尝试从最新的权重自动恢复。                                                                                  |
| `--amp`                               | 启用自动混合精度训练。                                                                                                                                              |
| `--no-validate`                       | **不建议使用**。在训练期间禁用权重评估。                                                                                                                            |
| `--auto-scale-lr`                     | 根据实际批次大小和原始批次大小自动缩放学习率。                                                                                                                      |
| `--seed`                              | 随机种子。                                                                                                                                                          |
| `--diff-rank-seed`                    | 是否为不同的 rank 设置不同的种子。                                                                                                                                  |
| `--deterministic`                     | 是否为 CUDNN 后端设置确定性选项。                                                                                                                                   |
| `--cfg-options CFG_OPTIONS`           | 覆盖使用的配置中的某些设置，xxx=yyy 格式的键值对将合并到配置文件中。如果要覆盖的值是一个列表，则应采用 `key="[a,b]"` 或 `key=a,b` 的形式。该参数还允许嵌套的列表/元组值，例如 `key="[(a,b),(c,d)]"`。请注意，引号是必需的，且不允许有空格。 |
| `--launcher {none,pytorch,slurm,mpi}` | 作业启动器的选项。默认为 `none`。                                                                                                                                   |

### 使用多个 GPU 进行训练

我们提供了一个 shell 脚本使用 `torch.distributed.launch` 来启动多个 GPU 的训练任务。

```shell
bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```

| 参数       | 描述                                                                    |
| ---------- | ----------------------------------------------------------------------- |
| `CONFIG`   | 配置文件的路径。                                                        |
| `GPUS`     | 要使用的 GPU 数量。                                                     |
| `[PYARGS]` | `tools/train.py` 的其他可选参数，请参见[这里](#使用单个-gpu-进行训练)。 |

您还可以通过环境变量来指定启动器的其他参数。例如，使用以下命令将启动器的通信端口更改为 29666：

```shell
PORT=29666 bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```

如果您想启动多个训练作业并使用不同的 GPU，可以通过指定不同的端口和可见设备来启动它们。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG} 4 [PY_ARGS]
```

### 使用多台机器进行训练

#### 同一网络中的多台机器

如果您使用以太网连接的多台机器启动训练作业，可以运行以下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

需要指定以下额外的环境变量来训练或测试多台机器上的模型：

| ENV_VARS      | 描述                                                             |
| ------------- | ---------------------------------------------------------------- |
| `NNODES`      | 机器的总数。默认为 1。                                           |
| `NODE_RANK`   | 本地机器的索引。默认为 0。                                       |
| `PORT`        | 通信端口，在所有机器上应该保持一致。默认为 29500。               |
| `MASTER_ADDR` | 主机器的 IP 地址，在所有机器上应该保持一致。默认为 `127.0.0.1`。 |

通常，如果您没有高速网络（如 InfiniBand），则速度会比较慢。

#### 使用 slurm 管理的多台机器

如果您在使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMAction2，可以使用脚本 `slurm_train.sh`。

```shell
[ENV_VARS] bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} [PY_ARGS]
```

下面是该脚本的参数描述。

| 参数        | 描述                                                                    |
| ----------- | ----------------------------------------------------------------------- |
| `PARTITION` | 集群中要使用的分区。                                                    |
| `JOB_NAME`  | 作业的名称，您可以自定义。                                              |
| `CONFIG`    | 配置文件的路径。                                                        |
| `[PYARGS]`  | `tools/train.py` 的其他可选参数，请参见[这里](#使用单个-gpu-进行训练)。 |

下面列出了可用于配置 slurm 作业的环境变量。

| ENV_VARS        | 描述                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `GPUS`          | 要使用的 GPU 数量。默认为 8。                                                    |
| `GPUS_PER_NODE` | 每个节点要分配的 GPU 数量。默认为 8。                                            |
| `CPUS_PER_TASK` | 每个任务要分配的 CPU 数量（通常一个 GPU 对应一个任务）。默认为 5。               |
| `SRUN_ARGS`     | `srun` 的其他参数。可用选项可在[这里](https://slurm.schedmd.com/srun.html)找到。 |

## 测试

### 使用单个 GPU 进行测试

您可以使用 `tools/test.py` 在一台带有 CPU 和可选 GPU 的单机上测试模型。

下面是脚本的完整用法：

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

````{note}
默认情况下，MMAction2 更倾向于使用 GPU 而不是 CPU 进行测试。如果您想在 CPU 上测试模型，请清空 `CUDA_VISIBLE_DEVICES` 或将其设置为 -1 以使 GPU 对程序不可见。

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```
````

| 参数                                  | 描述                                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件的路径。                                                                                                                                                    |
| `CHECKPOINT_FILE`                     | 权重文件的路径（可以是 HTTP 链接）。                                                                                                                                |
| `--work-dir WORK_DIR`                 | 保存包含评估指标的文件的目录。默认为与配置文件相同名称的文件夹，位于 `./work_dirs` 下。                                                                             |
| `--dump DUMP`                         | 存储模型的所有输出以进行离线评估的路径。                                                                                                                            |
| `--cfg-options CFG_OPTIONS`           | 覆盖使用的配置中的某些设置，xxx=yyy 格式的键值对将合并到配置文件中。如果要覆盖的值是一个列表，则应采用 `key="[a,b]"` 或 `key=a,b` 的形式。该参数还允许嵌套的列表/元组值，例如 `key="[(a,b),(c,d)]"`。请注意，引号是必需的，且不允许有空格。 |
| `--show-dir SHOW_DIR`                 | 保存结果可视化图片的目录。                                                                                                                                          |
| `--show`                              | 在窗口中可视化预测结果。                                                                                                                                            |
| `--interval INTERVAL`                 | 可视化的样本间隔。默认为 1。                                                                                                                                        |
| `--wait-time WAIT_TIME`               | 每个窗口的显示时间（单位：秒）。默认为 2。                                                                                                                          |
| `--launcher {none,pytorch,slurm,mpi}` | 作业启动器的选项。默认为 `none`。                                                                                                                                   |

### 使用多个 GPU 进行测试

我们提供了一个 shell 脚本使用 `torch.distributed.launch` 来启动多个 GPU 的测试任务。

```shell
bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} [PY_ARGS]
```

| 参数         | 描述                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| `CONFIG`     | 配置文件的路径。                                                       |
| `CHECKPOINT` | 权重文件的路径（可以是 HTTP 链接）。                                   |
| `GPUS`       | 要使用的 GPU 数量。                                                    |
| `[PYARGS]`   | `tools/test.py` 的其他可选参数，请参见[这里](#使用单个-gpu-进行测试)。 |

您还可以通过环境变量来指定启动器的其他参数。例如，使用以下命令将启动器的通信端口更改为 29666：

```shell
PORT=29666 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} [PY_ARGS]
```

如果您想启动多个测试作业并使用不同的 GPU，可以通过指定不同的端口和可见设备来启动它们。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 [PY_ARGS]
```

### 使用多台机器进行测试

#### 同一网络中的多台机器

如果您使用以太网连接的多台机器进行测试作业，可以运行以下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS
```

在第二台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS
```

与单台机器上的多个 GPU 相比，您需要指定一些额外的环境变量：

| ENV_VARS      | 描述                                                             |
| ------------- | ---------------------------------------------------------------- |
| `NNODES`      | 机器的总数。默认为 1。                                           |
| `NODE_RANK`   | 本地机器的索引。默认为 0。                                       |
| `PORT`        | 通信端口，在所有机器上应该保持一致。默认为 29500。               |
| `MASTER_ADDR` | 主机器的 IP 地址，在所有机器上应该保持一致。默认为 `127.0.0.1`。 |

通常，如果您没有高速网络（如 InfiniBand），则速度会比较慢。

#### 使用 slurm 管理的多台机器

如果您在使用 [slurm](https://slurm.schedmd.com/) 管理的集群上运行 MMAction2，可以使用脚本 `slurm_test.sh`。

```shell
[ENV_VARS] bash tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${CHECKPOINT} [PY_ARGS]
```

下面是该脚本的参数描述。

| 参数         | 描述                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| `PARTITION`  | 集群中要使用的分区。                                                   |
| `JOB_NAME`   | 作业的名称，您可以自定义。                                             |
| `CONFIG`     | 配置文件的路径。                                                       |
| `CHECKPOINT` | 权重文件的路径（可以是 HTTP 链接）。                                   |
| `[PYARGS]`   | `tools/test.py` 的其他可选参数，请参见[这里](#使用单个-gpu-进行测试)。 |

下面列出了可用于配置 slurm 作业的环境变量。

| ENV_VARS        | 描述                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `GPUS`          | 要使用的 GPU 数量。默认为 8。                                                    |
| `GPUS_PER_NODE` | 每个节点要分配的 GPU 数量。默认为 8。                                            |
| `CPUS_PER_TASK` | 每个任务要分配的 CPU 数量（通常一个 GPU 对应一个任务）。默认为 5。               |
| `SRUN_ARGS`     | `srun` 的其他参数。可用选项可在[这里](https://slurm.schedmd.com/srun.html)找到。 |
