-1. Download `opencode_nodata.zip`, then extract the code after decompression.

0. Go to https://github.com/facebookresearch/jepa and download VIT-L from Pretrained models. Place it in the checkpoints folder.

1. Run `0generate_CSV.py` #Generates CSV files conforming to the code requirements for the data.
2. Run `1trainOCTbranch.py` #Trains the OCT branch, saving epochs 18-25 to the checkpoints/weights folder.
3. Run `1trainCFPbranch.py` #Trains the CFP branch, saving results to the CFPcheckpoints folder.
4. Run `2val.py` and `2test.py` sequentially. #Find the optimal threshold for testing from the best threshold in `2val.py`.
5. Replace the following in `config_gk2` and `config_gk3`:
  `encoder_weights: "checkpoints/weights/encoder_epoch_20.pth"`
  `classifier_weights: "checkpoints/weights/classifier_epoch_20.pth"`
with
  `encoder_weights: "checkpoints/weights/encoder_epoch_21.pth"`
  `classifier_weights: "checkpoints/weights/classifier_epoch_21.pth"`
and so on...
6. Run `2val.py` and `2test.py` sequentially. #Find the optimal threshold for testing from the best threshold in `2val.py`.
7. Fill in the optimal thresholds within `cascade_thresholds [n,n,n]` in `configs_gk3` (Note: fill in the thresholds in the order output in step 3, which is non2, non0, non1).
8. Run `3test.py` #Obtain the inference results on the test set.

---------------------------------------------------------------------------------------------------------------------------------------------------
Due to the randomness introduced by distributed computing (even though we only used a single GPU and did not use distributed computing), the best kappa score is between 0.886-0.895 for step 2. To obtain the results in the paper, you can change the following lines in the `config_gk2` and `config_gk3` configuration files when running steps 3 and 5:

  `encoder_weights: "checkpoints/weights/encoder_epoch_25.pth"`
  `classifier_weights: "checkpoints/weights/classifier_epoch_25.pth"`
to
  `encoder_weights: "checkpoints/paperweights/encoder_epoch_25.pth"`
  `classifier_weights: "checkpoints/paperweights/classifier_epoch_25.pth"`
---------------------------------------------------------------------------------------------------------------------------------------------------
Confusion Matrix Generation:

1. Run `4hunxiao.py` #Generates the test CSV file.
2. Run `5schunxiao.py` #Generates the confusion matrix for the test set.
3. In `4hunxiao.py`, change `params['data']['dataset_val'] = os.path.join(current_directory, 'test_video.csv')` to `params['data']['dataset_val'] = os.path.join(current_directory, 'val_video.csv')`.
4. Run `4hunxiao.py` #Generates the val CSV file.
5. Run `5schunxiao.py` #Generates the confusion matrix for the val set.
6. Run `6生成融合前混淆矩阵` #Generates two confusion matrices before fusion.

---------------------------------------------------------------------------------------------------------------------------------------------------

Table 2:

1. Run `7消融实验table2第一行.py` #Obtain the results for the first row.
2. Run `7消融实验table2第二行.py` #Obtain the results for the second row.

---------------------------------------------------------------------------------------------------------------------------------------------------
Table 3:

1. Modify `configs_table2_2` to change `tubelet_size: 2` to 1 and 4.
2. Run `7消融实验table2第二行.py` separately for `tubelet_size: 1` and `tubelet_size: 4`.
---------------------------------------------------------------------------------------------------------------------------------------------------
Table 4:

1. Run `7消融实验table4.py` #Obtain the results for VIT.
2. Modify `configs_table4` to replace `timm_transformer_name: vit_large_patch14_dinov2.lvd142m` with the content in the comments.
3. Run `7消融实验table4.py`.

MAE Baseline	vit_base_patch16_mae

DINOv2 Baseline	vit_base_patch14_dinov2

CLIP ViT-L	vit_large_patch14_clip_224
---------------------------------------------------------------------------------------------------------------------------------------------------
Table 6:

1. In `1trainCFPbranch.py`, change `SELECTED_MODEL = "efficientnet_b3"` to the corresponding model.
2. Run `1trainCFPbranch.py`.
3. In `config_gk2` and `config_gk22`, change `best_efficientnet_b3` to the corresponding model.
4. Run `2val.py` and `2test.py` sequentially. #Find the optimal threshold for testing from the best threshold in `2val.py`, and the training results.
---------------------------------------------------------------------------------------------------------------------------------------------------
Table 7:

1. Run `7消融实验table7-2.py` #Obtain the LoRA training results.
2. In `config_gk2` and `config_gk22`, change
  `encoder_weights: "checkpoints/weights/encoder_epoch_25.pth"`
  `classifier_weights: "checkpoints/weights/classifier_epoch_25.pth"`
to the corresponding training results.
3. Run `2val.py` and `2test.py` sequentially. #Find the optimal threshold for testing from the best threshold in `2val.py`, and the training results.
---------------------------------------------------------------------------------------------------------------------------------------------------
Figures 4 and 5:

Run `10CAM.py` and modify `target_label` in `evals\video_classification_frozen\CAM.py` to 0, 1, and 2 to generate histograms for different labels.
