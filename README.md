# DL_SCAN_train
# USER GUIDE
## :rocket: About this project
### This is the project for the Central Weather Bureau (Taiwan) which predicts the initiation of convective cell using Deep learning Model.
### Follow the steps to run the model:
:bulb:
- [X] Make sure you prepare the SCAN&Radar data first
- [X] Simply run the pl_run.py to start training
- [X] Model checkpoint will be saved in /**root**/**user**/SCAN_checkpoints/
- [X] You can also track the training process by logger which is saved in /**root**/**user**/logs with tensorboard
- [X] Utilize the checkpoint to testing the result in another repository **/DLSCAN_eval**
- Contact *r07229013@ntu.edu.tw* if you have any problems

:rocket:
## Model structure
### This is our DLSCAN model schematic diagram. We utilize CNN layer combined with conv-GRU to form a U-NET structure. Then post-process the result with some tricky method (Interesing details will be in our ongoing paper).
![](https://i.imgur.com/xD2CcM4.png)

## Post-processing 
### This shows one of our post-processing down-scaling method with Maxpool
![](https://i.imgur.com/8O5U8Vp.png)

>Done on 15, Spetember, 2022 [name=peterpan]