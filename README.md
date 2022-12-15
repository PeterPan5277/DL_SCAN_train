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
## Input data
### This is our input data, we use radar CV and convection cell location 1 hour before
![image](https://user-images.githubusercontent.com/91505593/207801510-75654c9c-3e93-4e05-a76b-dfc7a626a7f0.png)


## Model structure
### This is our DLSCAN model schematic diagram. We utilize CNN layer combined with conv-GRU to form a U-NET structure. Then post-process the result with some tricky method (Interesing details will be in our ongoing paper).
![image](https://user-images.githubusercontent.com/91505593/207801661-a4d0e4ad-f964-42fd-a6ad-066b2acf3fdb.png)


## Post-processing 
### This shows one of our post-processing down-scaling method with Maxpool
![](https://i.imgur.com/8O5U8Vp.png)

## Final prediction 
### Here we show the final prediction with and without mask
![image](https://user-images.githubusercontent.com/91505593/207801341-c8000b09-0ca3-40cf-88fc-010a770c0165.png)


>Done on 15, Spetember, 2022 [name=peterpan]
