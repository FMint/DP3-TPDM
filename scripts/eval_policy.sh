# use the same command as training except the script
# for example:

# bash scripts/eval_policy.sh simple_dp3_ppo adroit_pen 0103 0 0

# 使用TPM模型评估（第6个参数是TPM模型路径）:
# bash scripts/eval_policy.sh simple_dp3_ppo adroit_door 0401_tpm 0 0 "checkpoints/tpm/tpm_20250716_143052/tpm_best_val.pt"

# 参数说明:
# $1: 算法名称 (alg_name)
# $2: 任务名称 (task_name) 
# $3: 附加信息 (addition_info)
# $4: 随机种子 (seed)
# $5: GPU ID (gpu_id)
# $6: TPM模型检查点路径 (可选)

DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/0717/${exp_name}_seed${seed}"

# 第6个参数：TPM模型检查点路径（可选）
tpm_ckpt=${6}

gpu_id=${5}


cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 如果提供了预训练模型路径，添加到命令中
if [ ! -z "$pretrained_ckpt" ]; then
    echo -e "\033[33mUsing pretrained model checkpoint: ${pretrained_ckpt}\033[0m"
    python eval_tpm.py --config-name=${config_name}.yaml \
                                task=${task_name} \
                                hydra.run.dir=${run_dir} \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device=cuda:0 \
                                exp_name=${exp_name} \
                                "training.tpm_checkpoint=${tpm_ckpt}"
else
    python eval.py --config-name=${config_name}.yaml \
                                task=${task_name} \
                                hydra.run.dir=${run_dir} \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device=cuda:0 \
                                exp_name=${exp_name}
fi



                                