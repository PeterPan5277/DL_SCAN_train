import argparse
import os, sys
import socket, pickle
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
os.environ['CUDA_VISIBLE_DEVICES']="0" #指定要用哪顆GPU
os.environ['ROOT_DATA_DIR']='/bk2/handsomedong/DLRA_database/'
path = os.getcwd()
path = path.split('/')[:-1] 


import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.cpp_extension import CUDA_HOME
from pytorch_lightning.callbacks import EarlyStopping

from core.data_loader_type import DataLoaderType
from core.enum import DataType
from core.loss_type import BlockAggregationMode, LossType
from core.model_type import ModelType
from data_loaders.pl_data_loader_module import PLDataLoader
from utils.run_utils import get_model, parse_date_end, parse_date_start, parse_dict

def main(args):
    
    epochs = 10
    workers = 8
    input_shape = (120, 120)
    loss_type = int(args.loss_kwargs.get('type', LossType.BCEwithlogits))#此處更改loss type!
    loss_aggregation_mode = int(args.loss_kwargs.get('aggregation_mode', BlockAggregationMode.MAX))
    loss_kernel_size = int(args.loss_kwargs.get('kernel_size', 5))
    residual_loss = bool(int(args.loss_kwargs.get('residual_loss', 0)))
    mixing_weight = float(args.loss_kwargs.get('w', 1))

    loss_kwargs = {
        'type': loss_type,
        'aggregation_mode': loss_aggregation_mode,
        'kernel_size': loss_kernel_size,
        'residual_loss': residual_loss,
        'w': mixing_weight
    }
    if loss_type in [LossType.SSIMBasedLoss, LossType.NormalizedSSIMBasedLoss]:
        loss_kwargs['mae_w'] = float(args.loss_kwargs.get('mae_w', 0.1))
        loss_kwargs['ssim_w'] = float(args.loss_kwargs.get('ssim_w', 0.02))

    data_kwargs = {
        'data_type': int(args.data_kwargs.get('type', DataType.Radar+DataType.Scan)), #這邊要改INPUTS 改成DataType.Scan or DataType.Radar
        'residual': bool(int(args.data_kwargs.get('residual', 0))),
        'target_offset': int(args.data_kwargs.get('target_offset', 0)),
        'target_len': int(args.data_kwargs.get('target_len', 1)), #SCAN targetlens=1
        'input_len': int(args.data_kwargs.get('input_len', 6)),
        'hourly_data': bool(int(args.data_kwargs.get('hourly_data', 0))), #######若在此加一些變化，也要在pl_data_loader_module/run_utils/dataloader改############
        'SCAN_data' : bool(int(args.data_kwargs.get('SCAN_data', 1))),     #SCAN_data
        'maxpool_atlast' : bool(int(args.data_kwargs.get('maxpool_atlast', 1))),  #maxpool at output or not
        'inp_moreSCAN' : bool(int(args.data_kwargs.get('inp_moreSCAN', 1))), #在初始INP就先加密
        'hetero_data': bool(int(args.data_kwargs.get('hetero_data', 0))),  #ERA5
        'sampling_rate': int(args.data_kwargs.get('sampling_rate', 5)), #在訓練時，每個只取一次#################################
        'prior_dtype': int(args.data_kwargs.get('prior', DataType.NoneAtAll)),
        'random_std': int(args.data_kwargs.get('random_std', 0)),
        'ith_grid': int(args.data_kwargs.get('ith_grid', -1)),
        'pad_grid': int(args.data_kwargs.get('pad_grid', 10)),
        'threshold': float(args.data_kwargs.get('threshold', 0.5)),
    }
    model_kwargs = {
        'adv_w': float(args.model_kwargs.get('adv_w', 0.01)),
        'model_type': ModelType.from_name(args.model_kwargs.get('type', 'Adv_SCAN')),
        'dis_d': int(args.model_kwargs.get('dis_d', 3)),
        'teach_force':float(args.model_kwargs.get('teach_force', 0)),
    }

    dm = PLDataLoader(
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        img_size=input_shape,
        dloader_type=args.dloader_type,
        **data_kwargs,
        batch_size=args.batch_size,
        num_workers=workers,
    )

    model = get_model(
        args.train_start,
        args.train_end,
        model_kwargs,
        loss_kwargs,
        data_kwargs,
        args.checkpoints_path,
        args.log_dir,
        data_loader_info=dm.model_related_info,
    )
    #logger = TensorBoardLogger(save_dir='logs', name=ModelType.name(model_kwargs['model_type']))
    #這是為了能夠從tensorboard觀察訓練過程，在model中可以用self.log()把要觀察的對象放入
    #若要看$tensorboard --logdir=/wk171/peterpan/logs/...
    #一次看兩個並命名tensorboard --logdir_spec=name1:./version_143,name2:./version_142
    logger = TestTubeLogger(save_dir='/wk171/peterpan/logs', name=ModelType.name(model_kwargs['model_type']))
    logger.experiment.argparse(args)
    logger.experiment.tag({'input_len': data_kwargs['input_len'], 'target_len': data_kwargs['target_len']})
    
    checkpoint_callback = model.get_checkpoint_callback()
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer.from_argparse_args(args, 
                                         gpus=1,
                                         max_epochs=20, 
                                         fast_dev_run=False,  
                                         logger=logger,
                                         benchmark= True, #若每次輸入大小皆相同，會較快
                                         callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, dm)  #.fit同時做了train and validation
    #default max epochs for pl is 1000

if __name__ == '__main__':
    # python scripts/pl_run.py --train_start=20150101 --train_end=20150131 --val_start=20150201 --val_end=20150331 --gpus=1 --batch_size=2 --loss_kwargs=type:1,kernel:10,aggregation_mode:1 --data_kwargs=sampling_rate:3,hetero_data:0 --precision=16 --model_type=BaselineCNN
    # python pl_run.py --batch_size=32 --data_kwargs=sampling_rate:3,hetero_data:0,target_len:3 --model_kwargs=type:BalancedGRUAdvPONI,teach_force:0.5
    print(socket.gethostname(), datetime.now().strftime("%y-%m-%d-%H:%M:%S"))
    print('Python version', sys.version)
    print('CUDA_HOME', CUDA_HOME)
    print('CudaToolKit Version', torch.version.cuda)
    print('torch Version', torch.__version__)
    print('torchvision Version', torchvision.__version__)

    parser = argparse.ArgumentParser() #--代表是全名/-代表簡易名
    # 若在命令列有輸入值，才會進行「type」的運算（預設string）；若無，直接回傳default
    parser.add_argument('--dloader_type', type=DataLoaderType.from_name, default=DataLoaderType.Native)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_start', type=parse_date_start, default=datetime(2015, 1, 1))
    parser.add_argument('--train_end', type=parse_date_end, default=datetime(2018, 12, 31, 23, 50))
    parser.add_argument('--val_start', type=parse_date_start, default=datetime(2019, 1, 1))
    parser.add_argument('--val_end', type=parse_date_end, default=datetime(2021, 12, 31, 23, 50))
    parser.add_argument('--loss_kwargs', type=parse_dict, default={})
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_kwargs', type=parse_dict, default={})
    parser.add_argument('--model_kwargs', type=parse_dict, default={})
    parser.add_argument('--checkpoints_path',
                        type=str,
                        default=('/wk171/peterpan/SCAN_checkpoints/'),
                        help='Full path to the directory where model checkpoints are [to be] saved')
    # 加入Trainer參數
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    #print(type(args.loss_kwargs)) dict
    main(args)