#!/bin/sh
#$ -V -cwd
#$ -l h_vmem=5G
python test1.py --name two_ouput_neurons -r True
