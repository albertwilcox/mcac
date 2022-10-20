# Note: Add a `--gen-data` flag the first time using any particular environment

################################
#     Pointmass Navigation     #
################################

# SAC
python scripts/train.py --algo sac --env navigation --n-demos 20 --tau 0.05 --init-iters 2000

# SAC MCAC
python scripts/train.py --algo sac --env navigation --n-demos 20 --tau 0.05 --init-iters 2000 --do-mcac-bonus

# TD3
python scripts/train.py --algo td3 --env navigation --n-demos 20 --init-iters 2000 --tau 0.05

# TD3 MCAC
python scripts/train.py --algo td3 --env navigation --n-demos 20 --init-iters 2000 --tau 0.05 --do-mcac-bonus

# GQE
python scripts/train.py --algo gqe --env navigation --n-demos 20 --tau 0.05 --total-timesteps 150000 --gqe --gqe-lambda 0.9

# GQE MCAC
python scripts/train.py --algo gqe --env navigation --n-demos 20 --tau 0.05 --total-timesteps 150000 --gqe --gqe-lambda 0.9 --do-mcac-bonus

# OEFD
python scripts/train.py --algo td3 --env navigation --n-demos 20 --init-iters 2000 --tau 0.05 --do-bc-loss --do-q-filter

# OEFD MCAC
python scripts/train.py --algo td3 --env navigation --n-demos 20 --init-iters 2000 --tau 0.05 --do-bc-loss --do-q-filter --do-mcac-bonus

# AWAC
python scripts/train.py --algo awac --env navigation --n-demos 20 --update-n-steps 50 --num-eval-episodes 10 --init-iters 10000

# AWAC MCAC
python scripts/train.py --algo awac --env navigation --n-demos 20 --update-n-steps 50 --num-eval-episodes 10 --init-iters 10000 --do-mcac-bonus

# CQL
python scripts/train.py --algo cql --env navigation --n-demos 20 --init-iters 10000 --num-eval-episodes 5 

# CQL MCAC
python scripts/train.py --algo cql --env navigation --n-demos 20 --init-iters 10000 --num-eval-episodes 5 --do-mcac-bonus



########################
#   Block Extraction   #
########################

# SAC
python scripts/train.py --algo sac --env extraction --n-demos 50 --num-eval-episodes 5 --alpha 0.1

# SAC MCAC
python scripts/train.py --algo sac --env extraction --n-demos 50 --num-eval-episodes 5 --alpha 0.1 --do-mcac-bonus

# TD3
python scripts/train.py --algo td3 --env extraction --n-demos 50 --num-eval-episodes 5

# TD3 MCAC
python scripts/train.py --algo td3 --env extraction --n-demos 50 --num-eval-episodes 5 --do-mcac-bonus

# GQE
python scripts/train.py --algo gqe --env extraction --alpha 0.1 --num-eval-episodes 5 --n-demos 50 --total-timesteps 200000 --gqe --gqe-n 8 --gqe-lambda 0.95

# GQE MCAC
python scripts/train.py --algo gqe --env extraction --alpha 0.1 --num-eval-episodes 5 --n-demos 50 --total-timesteps 200000 --do-mcac-bonus --gqe --gqe-n 8 --gqe-lambda 0.95

# OEFD
python scripts/train.py --algo td3 --env extraction --n-demos 50 --num-eval-episodes 5 --do-bc-loss --do-q-filter

# OEFD MCAC
python scripts/train.py --algo td3 --env extraction --n-demos 50 --num-eval-episodes 5 --do-bc-loss --do-q-filter --do-mcac-bonus

# AWAC
python scripts/train.py --algo awac --env extraction --n-demos 50 --update-n-steps 50 --num-eval-episodes 5 --init-iters 10000

# AWAC MCAC
python scripts/train.py --algo awac --env extraction --n-demos 50 --update-n-steps 50 --num-eval-episodes 5 --init-iters 10000 --do-mcac-bonus

# CQL
python scripts/train.py --algo cql --env extraction --n-demos 50 --init-iters 10000 --num-eval-episodes 5 

# CQL MCAC
python scripts/train.py --algo cql --env extraction --n-demos 50 --init-iters 10000 --num-eval-episodes 5 --do-mcac-bonus


##########################
#   Sequential Pushing   #
##########################

# SAC
python scripts/train.py --algo sac --env push --n-demos 500 --num-eval-episodes 5 --alpha 0.1

# SAC MCAC
python scripts/train.py --algo sac --env push --n-demos 500 --num-eval-episodes 5 --alpha 0.1 --do-mcac-bonus

# TD3
python scripts/train.py --algo td3 --env push --n-demos 500 --num-eval-episodes 5

# TD3 MCAC
python scripts/train.py --algo td3 --env push --n-demos 500 --num-eval-episodes 5 --do-mcac-bonus

# GQE
python scripts/train.py --algo gqe --env push --n-demos 500 --alpha 0.1 --num-eval-episodes 5 --total-timesteps 250000 --gqe --gqe-n 16 --gqe-lambda 0.95

# GQE MCAC
python scripts/train.py --algo gqe --env push --n-demos 500 --alpha 0.1 --num-eval-episodes 5 --total-timesteps 250000 --do-mcac-bonus --gqe --gqe-n 16 --gqe-lambda 0.95

# OEFD
python scripts/train.py --algo td3 --env push --n-demos 500 --num-eval-episodes 5 --do-bc-loss --do-q-filter

# OEFD MCAC
python scripts/train.py --algo td3 --env push --n-demos 500 --num-eval-episodes 5 --do-bc-loss --do-q-filter --do-mcac-bonus

# AWAC
python scripts/train.py --algo awac --env push --n-demos 500 --update-n-steps 50 --num-eval-episodes 5 --init-iters 10000

# AWAC MCAC
python scripts/train.py --algo awac --env push --n-demos 500 --update-n-steps 50 --num-eval-episodes 5 --init-iters 10000 --do-mcac-bonus

# CQL
python scripts/train.py --algo cql --env push --n-demos 500 --init-iters 10000 --num-eval-episodes 5 

# CQL MCAC
python scripts/train.py --algo cql --env push --n-demos 500 --init-iters 10000 --num-eval-episodes 5 --do-mcac-bonus



########################
#     Door Opening     #
########################

# SAC
python scripts/train.py --algo sac --env Door --n-demos 100 --alpha 0.05 --num-eval-episodes 5

# MCAC SAC
python scripts/train.py --algo sac --env Door --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --do-mcac-bonus

# TD3
python scripts/train.py --algo td3 --env Door --n-demos 100 --num-eval-episodes 5

# MCAC TD3
python scripts/train.py --algo td3 --env Door --n-demos 100 --num-eval-episodes 5 --do-mcac-bonus

# GQE
python scripts/train.py --algo gqe --env Door --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --total-timesteps 200000 --gqe --gqe-n 16

# GQE MCAC
python scripts/train.py --algo gqe --env Door --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --total-timesteps 200000 --do-mcac-bonus --gqe --gqe-n 16

# OEFD
python scripts/train.py --algo td3 --env Door --n-demos 100 --num-eval-episodes 5 --do-bc-loss --do-q-filter

# MCAC OEFD
python scripts/train.py --algo td3 --env Door --n-demos 100 --num-eval-episodes 5 --do-bc-loss --do-q-filter --do-mcac-bonus

# AWAC
python scripts/train.py --algo awac --env Door --n-demos 100 --num-eval-episodes 5 --update-n-steps 50 --init-iters 10000 --do-bc-loss --do-q-filter

# MCAC AWAC
python scripts/train.py --algo awac --env Door --n-demos 100 --num-eval-episodes 5 --update-n-steps 50 --init-iters 10000 --do-bc-loss --do-q-filter --do-mcac-bonus

# CQL
python scripts/train.py --algo cql --env Door --n-demos 100 --init-iters 10000 --num-eval-episodes 5 

# CQL MCAC
python scripts/train.py --algo cql --env Door --n-demos 100 --init-iters 10000 --num-eval-episodes 5 --do-mcac-bonus



########################
#     Block Lifting    #
########################

# SAC
python scripts/train.py --algo sac --env Lift --n-demos 100 --alpha 0.05 --num-eval-episodes 5

# MCAC SAC
python scripts/train.py --algo sac --env Lift --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --do-mcac-bonus

# TD3
python scripts/train.py --algo td3 --env Lift --n-demos 100 --num-eval-episodes 5

# MCAC TD3
python scripts/train.py --algo td3 --env Lift --n-demos 100 --num-eval-episodes 5 --do-mcac-bonus

# GQE
python scripts/train.py --algo gqe --env Lift --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --total-timesteps 200000 --gqe --gqe-n 16

# GQE MCAC
python scripts/train.py --algo gqe --env Lift --n-demos 100 --alpha 0.05 --num-eval-episodes 5 --total-timesteps 200000 --do-mcac-bonus --gqe --gqe-n 16

# OEFD
python scripts/train.py --algo td3 --env Lift --n-demos 100 --num-eval-episodes 5 --do-bc-loss --do-q-filter

# MCAC OEFD
python scripts/train.py --algo td3 --env Lift --n-demos 100 --num-eval-episodes 5 --do-bc-loss --do-q-filter --do-mcac-bonus

# AWAC
python scripts/train.py --algo awac --env Lift --n-demos 100 --num-eval-episodes 5 --update-n-steps 50 --init-iters 10000 --do-bc-loss --do-q-filter

# MCAC AWAC
python scripts/train.py --algo awac --env Lift --n-demos 100 --num-eval-episodes 5 --update-n-steps 50 --init-iters 10000 --do-bc-loss --do-q-filter --do-mcac-bonus

# CQL
python scripts/train.py --algo cql --env Lift --n-demos 100 --init-iters 10000 --num-eval-episodes 5 

# CQL MCAC
python scripts/train.py --algo cql --env Lift --n-demos 100 --init-iters 10000 --num-eval-episodes 5 --do-mcac-bonus

