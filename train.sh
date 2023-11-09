# dealing data
# nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity

# nnUNetv2_train 5 3d_fullres 1

# rm -rf ./nnUNet_results/Dataset005_OtherPet/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_best.pth
# for fold in 2 3 4
# do
#     nnUNetv2_train 6 3d_fullres $fold
#     rm -rf ./nnUNet_results/Dataset006_Kidney/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_$fold/checkpoint_best.pth
# 
# done

# running for 2d
# for fold in 0
# do
#     nnUNetv2_train 4 2d $fold
#     rm -rf ./nnUNet_results/Dataset004_Metastasis/nnUNetTrainer__nnUNetPlans__2d/fold_$fold/checkpoint_best.pth

# done

# for fold in 0
# do
#     nnUNetv2_train 4 2d $fold
#     rm -rf ./nnUNet_results/Dataset007_Kidney2d/nnUNetTrainer__nnUNetPlans__2d/fold_$fold/checkpoint_best.pth

# done