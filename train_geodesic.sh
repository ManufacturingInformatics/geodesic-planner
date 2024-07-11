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

echo "Number of episodes: ${episodes} | Repetition: ${rep}"
python3 ./geodesic-learner/train_geodesic.py -e ${episodes} -p ${path} -r ${rep}
echo "Plotting results..."
python3 ./geodesic-learner/plot_geodesic.py -r ${rep}