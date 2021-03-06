#!/bin/bash

python GEMSeismo.py --EMFull --ttinvar 1e-1 --invar_weight 1e-5 --fwdmodel GradBlur1.3 --model GradBlur1.3Blob3  --num_epochs 10 --num_subepochsE 800 --num_subepochsM 2000 --nsrc 49 --nrec 20  --btsize 32 --prior_sigma 1e-1 --data_sigma 1e-2 --phi_weight 1e-6 --Elr 1e-3 --Mlr 5e-5 --dv cuda:1  --px_randmean --use_eikonet --reduction sum --sine_activation --save_every 1000 --save_everyE 400 --surfaceR 1 --multidv -1   --dir FwdGrad
