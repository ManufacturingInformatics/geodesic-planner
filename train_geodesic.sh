#!/bin/sh
episodes=''
path=''
rep=''

while getopts e:p:r flag
do
    case "${flag}" in
        e) episodes=${OPTARG};;
        p) path=${OPTARG};;
        r) rep=${OPTARG};;
    esac
done

echo "Number of episodes: ${episodes}"
python3 ./geodesic-learner/train_geodesic.py -e ${episodes} -p ${path} -r ${rep}