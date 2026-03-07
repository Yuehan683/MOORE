@echo off

REM =====================================================
REM Activate conda environment
REM =====================================================
call conda activate algo_test

REM =====================================================
REM Go to MOORE project root
REM =====================================================
cd /d F:\moore\MOORE

REM =====================================================
REM Smoke test:
REM PPO + Multi-Task + PCGrad + MoE + MultiHead
REM =====================================================
python run_minigrid_ppo_mt_pcgrad.py ^
  --env_name MT5 ^
  --actor_network MixtureActor ^
  --critic_network MixtureCritic ^
  --n_experts 3 ^
  --n_head_features 64 64 64 64 64 ^
  --n_epochs 2 ^
  --n_steps 128 ^
  --seed 0

REM =====================================================
REM End
REM =====================================================
pause

