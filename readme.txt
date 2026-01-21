0.打开https://github.com/facebookresearch/jepa   下载Pretrained models中的VIT-L  放入checkpoints文件夹下



1.运行  0generate_CSV.py     #为数据生成符合代码要求的csv文件
2.运行  1trainOCTbranch.py                   #以训练oct分支，之后将会把第18-25个epoch保存至checkpoints/weights文件夹内
3.运行  1trainCFPbranch.py                    #以训练CPF分支，后会保存至CFPcheckpoints
3.先后运行  2val.py    和2test.py    #从2val中的最佳阈值中找到最适合test的  
4.替换config_gk2和config_gk3中的encoder_weights: "checkpoints/weights/encoder_epoch_20.pth"
  classifier_weights: "checkpoints/weights/classifier_epoch_20.pth"
至encoder_weights: "checkpoints/weights/encoder_epoch_21.pth"
  classifier_weights: "checkpoints/weights/classifier_epoch_21.pth"
等。。。。。。。
5.先后运行  2val.py    和2test.py    #从2val中的最佳阈值中找到最适合test的  
6.在configs_gk3中的cascade_thresholds [n,n,n]内填入最佳的阈值（注：按照第三步输出的阈值顺序填入，他的顺序是non2，non0，non1）            
7.运行  3test.py                     #得到测试集上的推理结果

---------------------------------------------------------------------------------------------------------------------------------------------------
步骤2由于分布式计算的原因（虽然我们并未使用分布式计算，只使用了一张GPU）随机性难以控制，最佳kappa在0.886-0.895之间，可以在运行步骤3和步骤5时更改配置文件config_gk2和config_gk3中的  encoder_weights: "checkpoints/weights/encoder_epoch_25.pth"
  classifier_weights: "checkpoints/weights/classifier_epoch_25.pth"
行为
  encoder_weights: "checkpoints/paperweights/encoder_epoch_25.pth"
  classifier_weights: "checkpoints/paperweights/classifier_epoch_25.pth"
以得到论文的结果
---------------------------------------------------------------------------------------------------------------------------------------------------
混淆矩阵生成
1.运行 4hunxiao.py   #以生成test的csv
2.运行 5schunxiao.py        #以生成test的混淆矩阵
3.更改  4hunxiao.py中    params['data']['dataset_val'] = os.path.join(current_directory, 'test_video.csv')    为    params['data']['dataset_val'] = os.path.join(current_directory, 'val_video.csv')
4.运行 4hunxiao.py        #以生成val的csv
5.运行 5schunxiao.py     #以生成test的混淆矩阵

6.运行  6生成融合前混淆矩阵                  #会生成两个融合前的混淆矩阵

---------------------------------------------------------------------------------------------------------------------------------------------------

表2
1.运行 7消融实验table2第一行.py   #获得第一行的结果
2.运行   7消融实验table2第二行.py   #获得第二行的结果

---------------------------------------------------------------------------------------------------------------------------------------------------
表3
1.修改配置文件  configs_table2_2  中  tubelet_size: 2，为1和4
2.分别运行  tubelet_size: 1 和  tubelet_size: 4的    7消融实验table2第二行.py 
---------------------------------------------------------------------------------------------------------------------------------------------------
表4
1.运行   7消融实验table4.py  获得vit
2.修改 configs_table4 中  timm_transformer_name: vit_large_patch14_dinov2.lvd142m为注释中的内容
3.运行   7消融实验table4.py  

MAE 基础版	vit_base_patch16_mae

DINOv2 基础版	vit_base_patch14_dinov2

CLIP ViT-L	vit_large_patch14_clip_224
---------------------------------------------------------------------------------------------------------------------------------------------------
表6
1.修改   1trainCFPbranch.py中SELECTED_MODEL = "efficientnet_b3"为对应模型
2.运行   1trainCFPbranch.py
3.修改config_gk2和config_gk22中的best_efficientnet_b3为对应模型
4.先后运行  2val.py    和2test.py    #从2val中的最佳阈值中找到最适合test的，以及训练结果
---------------------------------------------------------------------------------------------------------------------------------------------------
表7
1.运行  7消融实验table7-2.py 获得lora训练结果
2.修改   config_gk2和config_gk22中的  
  encoder_weights: "checkpoints/weights/encoder_epoch_25.pth"
  classifier_weights: "checkpoints/weights/classifier_epoch_25.pth"为对应训练结果
3.先后运行  2val.py    和2test.py    #从2val中的最佳阈值中找到最适合test的，以及训练结果
---------------------------------------------------------------------------------------------------------------------------------------------------
图4.5
运行10CAM,并修改
evals\video_classification_frozen\CAM.py中的target_label为0 1 2以做不同标签的直方图



