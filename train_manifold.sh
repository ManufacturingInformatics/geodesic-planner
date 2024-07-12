#!/bin/sh
mode=''
batch=''
num_datapoints=''

while getopts m:b:d: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
        b) batch_size=${OPTARG};;
        d) num_datapoints=${OPTARG};;
    esac
done

echo "Mode is ${mode} with batch size of ${batch_size} and ${num_datapoints} datapoints"
python3 ./manifold-learner/main.py --mode ${mode} --batch_size ${batch_size} --num_datapoints ${num_datapoints}