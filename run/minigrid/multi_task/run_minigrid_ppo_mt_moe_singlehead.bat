@echo off

cd /d %~dp0\..\..\..

set MT_NAME=%1
set N_EXPERTS=%2

python run_minigrid_ppo_mt.py --n_exp 30 ^
    --env_name %MT_NAME% ^
    --exp_name ppo_mt_moore_%MT_NAME%_%N_EXPERTS%e ^
    --n_epochs 100 --n_steps 2000 --n_episodes_test 16 --train_frequency 2000 ^
    --lr_actor 1e-3 --lr_critic 1e-3 ^
    --critic_network MiniGridPPOMixtureSHNetwork --critic_n_features 128 --n_experts %N_EXPERTS% ^
    --actor_network MiniGridPPOMixtureSHNetwork --actor_n_features 128 ^
    --batch_size 256 --gamma 0.99
    REM --wandb --wandb_entity your_entity
