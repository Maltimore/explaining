#!/bin/sh
#$ -V -cwd
#$ -l h_vmem=5G
#$ -e ./errors/$JOB_NAME
#$ -o ./text_output/$JOB_NAME
python test1.py --name test1 -r True
