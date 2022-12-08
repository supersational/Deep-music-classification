import os
for args in ['--tag vanilla',
'--l1 0.0 --tag no_l1',
'--dropout 0.0 --tag no_dropout',
'--alpha 0.0 --tag no_alpha',
'--alpha 0.0 --dropout --tag no_alpha_or_dropout',
]:
    for model in ['deep', 'shallow', 'filter']:
        os.system(f'python main.py --batch_size 1024 --model {model} --wandb {args}')
 