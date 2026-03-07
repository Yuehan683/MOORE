@echo off

cd /d %~dp0\..\..\..

set MT_NAME=%1
set N_EXPERTS=%2

python run_minigrid_ppo_mt_svd.py ^
    --n_exp 1 ^
    --env_name %MT_NAME% ^
    --exp_name smoke_test_moore_singlehead_svd_%N_EXPERTS%e ^
    --n_epochs 2 ^
    --n_steps 100 ^
    --train_frequency 100 ^
    --n_episodes_test 1 ^
    --lr_actor 1e-3 ^
    --lr_critic 1e-3 ^
    --critic_network MiniGridPPOMixtureSHNetwork_SVD ^
    --critic_n_features 64 ^
    --actor_network MiniGridPPOMixtureSHNetwork_SVD ^
    --actor_n_features 64 ^
    --n_experts %N_EXPERTS% ^
    --orthogonal ^
    --batch_size 64 ^
    --gamma 0.99
    REM --wandb