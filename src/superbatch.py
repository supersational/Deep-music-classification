import os
import re
import time
model = 'shallow'
for run_n in range(1,6):
    for run_type in [1,4]:
        print('run ',run_n)
        print('run_type',run_type)
        with open('train_auto.sh','w') as f:
            if run_type == 1:
                cmd = f'python -u main.py --epochs 200 --batch_size 4096 --tag default --model {model} --run_n {run_type}{run_n} >> "./slogs/log_bc4_$SLURM_JOB_ID.txt"'
            else:
                cmd = f'python -u main.py --epochs 200 --batch_size 4096 --dropout 0.0 --alpha 0.0 --tag ndnanl1 --l1 0.0  --model {model} --run_n {run_type}{run_n} >> "./slogs/log_bc4_$SLURM_JOB_ID.txt"'

            f.write(f"""#!/usr/bin/env bash

#SBATCH --job-name=lab4
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./slogs/log_bc4_%j.out # STDOUT out
#SBATCH -e ./slogs/log_bc4_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=16GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"
{cmd}
""")

        txt = os.popen('sbatch train_auto.sh').read()
        print(txt)
        print('='*100)
        time.sleep(0.5)