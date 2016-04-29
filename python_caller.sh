#!/bin/sh
#$ -V -cwd
#$ -l h_vmem=5G
python overall_script.py --name two_ouput_neurons -r True
