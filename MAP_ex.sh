#!/bin/bash

python MAPSeismo.py --multidv -1 --gridsrcs --ttinvar 1e2 --invar_weight 1e-2 --fwdmodel GradBlur1.3Grad --model GradBlur1.3Blob3 --num_epochs 5000 --nsrc 25 --nrec 20  --btsize 16 --prior_sigma 1e-1  --data_sigma 1e-2 --phi_weight 1e-1 --Elr 1e-4 --dv cuda:1  --px_randmean --use_eikonet --reduction sum --sine_activation --save_every 500 --multidv 1 3 --surfaceR 1 --pltRec --dir TEST
