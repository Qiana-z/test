
set -x

project_name='GRPO'
exp_name='GRPO-Qwen3-3b-gsm8k-fsdp2-one-step-off-2-6'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen/Qwen2.5-3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

NNODES=${NNODES:-1}  # 节点数量，默认1
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}  # 每个节点的GPU数量，默认8

n_gpus_rollout=2  # 用于rollout的GPU数量，默认2（vllm推理）
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))  # 剩下 6 张卡用来做训练（actor+critic+ref）

python3 -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=1152 \  # 全局训练 batch size，global_batch = per_gpu_batch * n_gpus * grad_accum
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \  # 用 FSDP v2 作为分布式训练策略，参数在多个 GPU 间 shard 分布，减显存
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \  # 指定 actor（policy）使用的初始模型权重路径。ref 模型通常会从这里 clone 一份。
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \  # 不使用「Hybrid Engine」（通常指训练和推理由一个引擎统一调度），这里 rollout 用 vLLM，训练用 HF/FSDP，两套组件分开。
    actor_rollout_ref.model.use_remove_padding=True \  # 启用「去 padding」优化：把实际 token 有效长度对齐成最小块，提高计算效率。
    actor_rollout_ref.actor.ppo_mini_batch_size=192 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \  # 一种 numerically stable 的 KL 实现
    actor_rollout_ref.actor.entropy_coeff=0 \  # 不额外鼓励策略熵（探索），比较「保守」
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \  # 不把 KL 加在 reward 上，只在 loss 中用 KL 约束（更符合 GRPO/DAPO 推荐实践）
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@