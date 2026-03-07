@echo off
cd /d %~dp0\..\..\..

set ENV_NAME=%1
set N_EXPERTS=%2
set LOAD_DIR=%3

python run_minigrid_ppo_tl.py --n_exp 1 --env_name %ENV_NAME% --exp_name smoke_test_tl_moore_multihead_%N_EXPERTS%e --n_epochs 2 --n_steps 200 --n_episodes_test 1 --train_frequency 200 --lr_actor 1e-3 --lr_critic 1e-3 --critic_network MiniGridPPOMixtureMHNetwork --critic_n_features 128 --orthogonal --n_experts %N_EXPERTS% --actor_network MiniGridPPOMixtureMHNetwork --actor_n_features 128 --batch_size 64 --gamma 0.99 --load_actor %LOAD_DIR% --load_critic %LOAD_DIR%
