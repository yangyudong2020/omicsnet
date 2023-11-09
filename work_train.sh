for fold in 0
do
    nnUNetv2_train 4 2d $fold
    rm -rf ./nnUNet_results/Dataset004_Metastasis/nnUNetTrainer__nnUNetPlans__2d/fold_$fold/checkpoint_best.pth

done