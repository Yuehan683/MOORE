@echo off
cd /d F:\moore\MOORE

set ENV_NAME=%1
set N_EXPERTS=%2

python run_minigrid_ppo_tl.py ^
 --n_exp 1 ^
 --env_name %ENV_NAME% ^
 --exp_name smoke_test_tl_moe_singlehead_%N_EXPERTS%e_noinit ^
 --n_epochs 2 ^
 --n_steps 500 ^
 --n_episodes_test 2 ^
 --train_frequency 500 ^
 --lr_actor 1e-3 ^
 --lr_critic 1e-3 ^
 --critic_network MiniGridPPOMixtureSHNetwork ^
 --critic_n_features 128 ^
 --n_experts %N_EXPERTS% ^
 --actor_network MiniGridPPOMixtureSHNetwork ^
 --actor_n_features 128 ^
 --batch_size 64 ^
 --gamma 0.99
