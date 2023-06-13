# Training and Test

## Training

### Training with your PC

You can use `tools/train.py` to train a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

````{note}
By default, MMAction2 prefers GPU to CPU. If you want to train a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program.

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [ARGS]
```
````

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `--work-dir WORK_DIR`                 | The target folder to save logs and checkpoints. Defaults to a folder with the same name of the config file under `./work_dirs`.                                     |
| `--resume [RESUME]`                   | Resume training. If a path is specified, resume from it, while if not specified, try to auto resume from the latest checkpoint.                                     |
| `--amp`                               | Enable automatic-mixed-precision training.                                                                                                                          |
| `--no-validate`                       | **Not suggested**. Disable checkpoint evaluation during training.                                                                                                   |
| `--auto-scale-lr`                     | Auto scale the learning rate according to the actual batch size and the original batch size.                                                                        |
| `--seed`                              | Random seed.                                                                                                                                                        |
| `--diff-rank-seed`                    | Whether or not set different seeds for different ranks.                                                                                                             |
| `--deterministic`                     | Whether to set deterministic options for CUDNN backend.                                                                                                             |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that the quotation marks are necessary and that no white space is allowed. |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher. Defaults to `none`.                                                                                                                       |

### Training with multiple GPUs

We provide a shell script to start a multi-GPUs task with `torch.distributed.launch`.

```shell
bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```

| ARGS       | Description                                                                        |
| ---------- | ---------------------------------------------------------------------------------- |
| `CONFIG`   | The path to the config file.                                                       |
| `GPUS`     | The number of GPUs to be used.                                                     |
| `[PYARGS]` | The other optional arguments of `tools/train.py`, see [here](#train-with-your-pc). |

You can also specify extra arguments of the launcher by environment variables. For example, change the
communication port of the launcher to 29666 by the following command:

```shell
PORT=29666 bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```

If you want to startup multiple training jobs and use different GPUs, you can launch them by specifying
different port and visible devices.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG} 4 [PY_ARGS]
```

### Training with multiple machines

#### Multiple machines in the same network

If you launch a training job with multiple machines connected with ethernet, you can run the following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

The following extra environment variables need to be specified to train or test models with multiple machines:

| ENV_VARS      | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `NNODES`      | The total number of machines. Defaults to 1.                                                          |
| `NODE_RANK`   | The index of the local machine. Defaults to 0.                                                        |
| `PORT`        | The communication port, it should be the same in all machines. Defaults to 29500.                     |
| `MASTER_ADDR` | The IP address of the master machine, it should be the same in all machines. Defaults to `127.0.0.1`. |

Usually it is slow if you do not have high speed networking like InfiniBand.

#### Multiple machines managed with slurm

If you run MMAction2 on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
[ENV_VARS] bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} [PY_ARGS]
```

Here are the arguments description of the script.

| ARGS        | Description                                                                        |
| ----------- | ---------------------------------------------------------------------------------- |
| `PARTITION` | The partition to use in your cluster.                                              |
| `JOB_NAME`  | The name of your job, you can name it as you like.                                 |
| `CONFIG`    | The path to the config file.                                                       |
| `[PYARGS]`  | The other optional arguments of `tools/train.py`, see [here](#train-with-your-pc). |

Here are the environment variables can be used to configure the slurm job.

| ENV_VARS        | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `GPUS`          | The number of GPUs to be used. Defaults to 8.                                                              |
| `GPUS_PER_NODE` | The number of GPUs to be allocated per node. Defaults to 8.                                                |
| `CPUS_PER_TASK` | The number of CPUs to be allocated per task (Usually one GPU corresponds to one task). Defaults to 5.      |
| `SRUN_ARGS`     | The other arguments of `srun`. Available options can be found [here](https://slurm.schedmd.com/srun.html). |

## Test

### Test with your PC

You can use `tools/test.py` to test a model on a single machine with a CPU and optionally a GPU.

Here is the full usage of the script:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

````{note}
By default, MMAction2 prefers GPU to CPU. If you want to test a model on CPU, please empty `CUDA_VISIBLE_DEVICES` or set it to -1 to make GPU invisible to the program.

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```
````

| ARGS                                  | Description                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | The path to the config file.                                                                                                                                        |
| `CHECKPOINT_FILE`                     | The path to the checkpoint file (It can be a http link)                                                                                                             |
| `--work-dir WORK_DIR`                 | The directory to save the file containing evaluation metrics. Defaults to a folder with the same name of the config file under `./work_dirs`.                       |
| `--dump DUMP`                         | The path to dump all outputs of the model for offline evaluation.                                                                                                   |
| `--cfg-options CFG_OPTIONS`           | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that the quotation marks are necessary and that no white space is allowed. |
| `--show-dir SHOW_DIR`                 | The directory to save the result visualization images.                                                                                                              |
| `--show`                              | Visualize the prediction result in a window.                                                                                                                        |
| `--interval INTERVAL`                 | The interval of samples to visualize. Defaults to 1.                                                                                                                |
| `--wait-time WAIT_TIME`               | The display time of every window (in seconds). Defaults to 2.                                                                                                       |
| `--launcher {none,pytorch,slurm,mpi}` | Options for job launcher. Defaults to `none`.                                                                                                                       |

### Test with multiple GPUs

We provide a shell script to start a multi-GPUs task with `torch.distributed.launch`.

```shell
bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} [PY_ARGS]
```

| ARGS         | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `CONFIG`     | The path to the config file.                                                     |
| `CHECKPOINT` | The path to the checkpoint file (It can be a http link)                          |
| `GPUS`       | The number of GPUs to be used.                                                   |
| `[PYARGS]`   | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc). |

You can also specify extra arguments of the launcher by environment variables. For example, change the
communication port of the launcher to 29666 by the following command:

```shell
PORT=29666 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} [PY_ARGS]
```

If you want to startup multiple test jobs and use different GPUs, you can launch them by specifying
different port and visible devices.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 [PY_ARGS]
```

### Test with multiple machines

#### Multiple machines in the same network

If you launch a test job with multiple machines connected with ethernet, you can run the following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPUS
```

Compared with multi-GPUs in a single machine, you need to specify some extra environment variables:

| ENV_VARS      | Description                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| `NNODES`      | The total number of machines. Defaults to 1.                                                          |
| `NODE_RANK`   | The index of the local machine. Defaults to 0.                                                        |
| `PORT`        | The communication port, it should be the same in all machines. Defaults to 29500.                     |
| `MASTER_ADDR` | The IP address of the master machine, it should be the same in all machines. Defaults to `127.0.0.1`. |

Usually it is slow if you do not have high speed networking like InfiniBand.

#### Multiple machines managed with slurm

If you run MMAction2 on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
[ENV_VARS] bash tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${CHECKPOINT} [PY_ARGS]
```

Here are the arguments description of the script.

| ARGS         | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `PARTITION`  | The partition to use in your cluster.                                            |
| `JOB_NAME`   | The name of your job, you can name it as you like.                               |
| `CONFIG`     | The path to the config file.                                                     |
| `CHECKPOINT` | The path to the checkpoint file (It can be a http link)                          |
| `[PYARGS]`   | The other optional arguments of `tools/test.py`, see [here](#test-with-your-pc). |

Here are the environment variables can be used to configure the slurm job.

| ENV_VARS        | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `GPUS`          | The number of GPUs to be used. Defaults to 8.                                                              |
| `GPUS_PER_NODE` | The number of GPUs to be allocated per node. Defaults to 8.                                                |
| `CPUS_PER_TASK` | The number of CPUs to be allocated per task (Usually one GPU corresponds to one task). Defaults to 5.      |
| `SRUN_ARGS`     | The other arguments of `srun`. Available options can be found [here](https://slurm.schedmd.com/srun.html). |
