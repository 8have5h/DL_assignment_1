#!/bin/bash
#PBS -N My_Job
#PBS -P col380.cs5210594.course
#PBS -q high
#PBS -l select=1:ncpus=1:ngpus=1:centos=skylake
#PBS -l walltime=00:02:00
python3 /home/cse/dualcs5210594/HPC-IIT-DELHI/COL775-Deep-Learning/assignments/Birds_25/a.py > /home/cse/dualcs5210594/HPC-IIT-DELHI/COL775-Deep-Learning/assignments/Birds_25/log.txt


