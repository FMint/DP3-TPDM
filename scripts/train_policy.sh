# Examples:
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0

# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0302 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_pen 0101 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_door 0201 0 0

# bash scripts/train_policy.sh simple_dp3_ppo adroit_door 0401 0 0

# 使用预训练模型训练TPM（第6个参数是预训练模型路径）:
# bash scripts/train_policy.sh simple_dp3_ppo adroit_door 0201 0 0 "data/outputs/0716/adroit_door-simple_dp3-0201_seed0/checkpoints/epoch-1400-test_mean_score-0.550.ckpt"

# 参数说明:
# $1: 算法名称 (alg_name)
# $2: 任务名称 (task_name) 
# $3: 附加信息 (addition_info)
# $4: 随机种子 (seed)
# $5: GPU ID (gpu_id)
# $6: 预训练模型检查点路径 (可选)

DEBUG=False
# DEBUG=True
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/0717/${exp_name}_seed${seed}"

# 第6个参数：预训练模型检查点路径（可选）
pretrained_ckpt=${6}

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    # wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 如果提供了预训练模型路径，添加到命令中
if [ ! -z "$pretrained_ckpt" ]; then
    echo -e "\033[33mUsing pretrained model checkpoint: ${pretrained_ckpt}\033[0m"
    python train_tpm.py --config-name=${config_name}.yaml \
                                task=${task_name} \
                                hydra.run.dir="${run_dir}" \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device=cuda:0 \
                                exp_name=${exp_name} \
                                logging.mode=${wandb_mode} \
                                checkpoint.save_ckpt=${save_ckpt} \
                                "training.main_model_checkpoint=${pretrained_ckpt}"
else
    python train.py --config-name=${config_name}.yaml \
                                task=${task_name} \
                                hydra.run.dir="${run_dir}" \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device=cuda:0 \
                                exp_name=${exp_name} \
                                logging.mode=${wandb_mode} \
                                checkpoint.save_ckpt=${save_ckpt}
fi



                                