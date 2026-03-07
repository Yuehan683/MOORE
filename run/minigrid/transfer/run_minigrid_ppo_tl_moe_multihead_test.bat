@echo off
set ENV_NAME=%1
set N_EXPERTS=%2

python F:\moore\MOORE\run_minigrid_ppo_tl.py ^
--env_name %ENV_NAME% ^
--exp_type TL ^
--actor_network MiniGridPPOMixtureMHNetwork ^
--critic_network MiniGridPPOMixtureMHNetwork ^
--n_experts %N_EXPERTS% ^
--orthogonal ^
--batch_size 8 ^
--train_frequency 16 ^
--n_epochs 2 ^
--n_steps 200 ^
--n_episodes_test 1 ^
--n_exp 1 ^
--lr_actor 3e-4 ^
--lr_critic 3e-4 ^
--gamma 0.99 ^
--gamma_eval 0.99 ^
--results_dir F:\moore\MOORE\results ^
--exp_name smoke_test_tl_moe_multihead_%N_EXPERTS%e ^
--debug

