ps -def | grep train.sh | cut -c 9-16 | xargs kill -9
ps -def | grep nnUNetv2_train | cut -c 9-16 | xargs kill -9