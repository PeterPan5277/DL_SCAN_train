import os
from collections import OrderedDict
from click import progressbar
from pyparsing import traceParseAction

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

import numpy as np
from core.running_average import RunningAverage
from models.base_model import BaseModel
from models.discriminator import Discriminator, Discriminator2, Discriminator3, Discriminator4
from models.loss import get_criterion
from utils.analysis_utils import DiscriminatorStats
from utils.performance_diagram import PerformanceDiagramStable
from core.loss_type import LossType

class Adv_SCAN(LightningModule):
    def __init__(self, adv_weight, discriminator_downsample, encoder, forecaster, ipshape, target_len, maxpool_atlast,
                 loss_kwargs, checkpoint_prefix, checkpoint_directory):
        super().__init__()
        # super().__init__(loss_kwargs, checkpoint_prefix, checkpoint_directory)
        self.encoder = encoder
        self.forecaster = forecaster
        self.G = torch.nn.Sequential(self.encoder, 
                                     self.forecaster)
        ''' jeffery add '''
        self._img_shape = ipshape
        ''' '''
        self.D = Discriminator(self._img_shape, downsample=discriminator_downsample)
        self._adv_w = adv_weight
        self.lr = 1e-04
        self._loss_type = loss_kwargs['type']
        self._ckp_prefix = checkpoint_prefix
        self._ckp_dir = checkpoint_directory
        self._target_len = target_len
        self._maxpool_atlast = maxpool_atlast# max pool at output last??
        self._criterion = get_criterion(loss_kwargs)
        self._recon_loss = RunningAverage()
        self._GD_loss = RunningAverage()
        self._G_loss = RunningAverage()
        self._D_loss = RunningAverage()
        self._focal_p = RunningAverage()####
        self._focal_n = RunningAverage()####
        self._p_len = 0
        self._n_len = 0
        self._p_val = 0
        self._n_val = 0
        self._label_smoothing_alpha = 0.001
        # self._train_D_stats = DiscriminatorStats()
        self._val_criterion = PerformanceDiagramStable()
        self._D_stats = DiscriminatorStats()
        print(f'[{self.__class__.__name__} W:{self._adv_w}] Ckp:{os.path.join(self._ckp_dir,self._ckp_prefix)} ')

    def adversarial_loss_fn(self, y_hat, y, smoothing=True):
        uniq_y = torch.unique(y)
        assert len(uniq_y) == 1
        if smoothing:
            # one sided smoothing.
            y = y * (1 - self._label_smoothing_alpha)
        return nn.BCELoss()(y_hat, y)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        #opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g

    def generator_loss(self, input_data, target_label):
        predicted_reconstruction = self(input_data)#1 16 120 120 ，若有最大池化-> 1 16 20 20
        #print(predicted_reconstruction.shape) #torch.Size([1, 16, 20, 20])
        #print(target_label.shape) #torch.Size([16, 1, 20, 20])
        predicted_reconstruction=predicted_reconstruction.permute((1,0,2,3))
        if self._loss_type==LossType.Focalloss:
            recons_loss = self._criterion(predicted_reconstruction, target_label, alpha=0.99, gamma=2,reduction='mean')
        else:
            recons_loss = self._criterion(predicted_reconstruction, target_label)
        p_idx=torch.where(target_label==1)#標出1的loc
        n_idx=torch.where(target_label==0)
        predp = predicted_reconstruction[p_idx]#答案為1時的預報
        predn = predicted_reconstruction[n_idx]
        self._p_len+= len(predp)
        self._n_len+= len(predn)
        tarp = target_label[p_idx]#答案為1
        tarn = target_label[n_idx]

        p_loss = nn.BCEWithLogitsLoss()(predp, tarp) 
        n_loss = nn.BCEWithLogitsLoss()(predn, tarn)
        #print('ploss in batch avg:', p_loss.item())
        #print('nloss in batch avg:', n_loss.item())
        #print('total in batch avg:', recons_loss.item())
        
        #p_loss = self._criterion(predp, tarp, alpha=0.25, gamma=3,reduction='mean')
        #n_loss = self._criterion(predn, tarn, alpha=0.25, gamma=3,reduction='mean')
        #代表最後要總平均輸出一個值，原本輸出16 1 120 120變成1
        #
        #改成沒有使用sigmoid在最後一層
        #recons_loss = nn.BCEWithLogitsLoss()(predicted_reconstruction, target_label)
        #print(type(recons_loss)) #torch tensor
        #print((recons_loss)) #1個值  if focal loss 16 1 120 120
        
        # train generator
        #N = self._target_len * input_data.size(0)
        #valid = torch.ones(N, 1)
        #valid = valid.type_as(input_data)

        # adversarial loss is binary cross-entropy
        # ReLU is used since generator fools discriminator with -ve values
        #adv_loss = self.adversarial_loss_fn(
        #    self.D(nn.ReLU()(predicted_reconstruction)).view(N, 1), valid, smoothing=False)
        #tqdm_dict = {'recon_loss': recons_loss, 'adv_loss': adv_loss}
        tqdm_dict = {'recon_loss': recons_loss,'Focal+': p_loss, 'Focal-': n_loss, 'l_p':len(predp), 'l_n':len(predn)}
        #loss = adv_loss * self._adv_w + recons_loss * (1 - self._adv_w)
        output = OrderedDict({'loss': recons_loss, 'progress_bar': tqdm_dict, 'prediction': predicted_reconstruction})
        return output

    def discriminator_loss(self, input_data, target_label):
        predicted_reconstruction = self(input_data)
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)
        d_out = self.D(target_label)
        real_loss = self.adversarial_loss_fn(d_out.view(N, 1), valid)

        # how well can it label as fake?
        fake = torch.zeros(N, 1)
        fake = fake.type_as(input_data)
        # ReLU is used since generator fools discriminator with -ve values
        # detach here is important fot not Back Propagating
        fake_loss = self.adversarial_loss_fn(self.D(nn.ReLU()(predicted_reconstruction.detach())).view(N, 1), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict})
        return output

    def training_step(self, batch, batch_idx):

        train_data, train_label = batch
        #print(train_data.shape)   #16 6 1 120 120 (batch, inp-sequence, rain/radar, x,y)
        #print(train_label.shape)  #16 1 120 120 (batch, opt-sequence, x,y)
        N = train_data.shape[0]
        
        loss_dict = self.generator_loss(train_data, train_label)
        l_p = loss_dict['progress_bar']['l_p']#length 代表120*120*16裡面有幾個+
        l_n = loss_dict['progress_bar']['l_n']#length 同上(有幾個-)
       
        #print('P_loss:',loss_dict['progress_bar']['Focal+'].item() * l_p) #.item()是torch tensor->一般純量
        #print('N_loss:',loss_dict['progress_bar']['Focal-'].item() * l_n)
        #print('TOTOAL_loss:',loss_dict['progress_bar']['recon_loss'].item() * N*120*120)
        #print('正樣本所占比例:',(loss_dict['progress_bar']['Focal+'].item() * l_p)/(loss_dict['progress_bar']['recon_loss'].item() * N*120*120)*100,'%')

        self._recon_loss.add(loss_dict['progress_bar']['recon_loss'].item() * N, N)
        self._focal_p.add(loss_dict['progress_bar']['Focal+'].item() * N, N)
        self._focal_n.add(loss_dict['progress_bar']['Focal-'].item() * N, N)
        #self._GD_loss.add(loss_dict['progress_bar']['adv_loss'].item() * N, N)
        #self._G_loss.add(loss_dict['loss'].item() * N, N)
        #self.log('G', self._G_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('Focal+', self._focal_p.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('Focal-', self._focal_n.get(), on_step=True, on_epoch=True, prog_bar=True)
        #print('正樣本所占比例:',self._focal_p.get()/(self._focal_n.get()+self._focal_p.get())*100,'%')
        #self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        # train discriminator
        #if optimizer_idx == 1:
        #    loss_dict = self.discriminator_loss(train_data, train_label, train_mask)
        #    self._D_loss.add(loss_dict['loss'].item() * N, N)

        #    self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict['loss']
    def training_epoch_end(self, training_step_outputs):
        #print('p_case in train:', self._p_len,'n_case in train',self._n_len)
        print(f'負樣本為正樣本的{round(self._n_len/self._p_len,0)}倍')
  
    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        # generator
    
        loss_dict = self.generator_loss(val_data, val_label)
        #aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        #self._val_criterion.compute(aligned_prediction, val_label)

        # Discriminator stats
        #pos = self.D(val_label)
        #neg = self.D(aligned_prediction)
        #self._D_stats.update(neg, pos)

        log_data = loss_dict.pop('progress_bar')
        return {'val_loss': log_data['recon_loss'], 'N': val_label.shape[0],
         'val_p_loss': log_data['Focal+'].item(), 'val_n_loss': log_data['Focal-'].item(),
         'p_len':log_data['l_p'], 'n_len':log_data['l_n']}

    def validation_epoch_end(self, outputs): #outputs是validation裡面所有step(一個step=一次batch)的outs
        val_loss_sum = 0
        val_focal_psum=0
        val_focal_nsum=0
        N = 0
        pl= 0
        nl= 0
        for output in outputs:
            val_loss_sum += output['val_loss'] * output['N']
            val_focal_psum+= output['val_p_loss'] * output['N']
            val_focal_nsum+= output['val_n_loss'] * output['N']
            # this may not have the entire batch. but we are still multiplying it by N
            N += output['N']
            pl+= output['p_len']
            nl+= output['n_len']

        val_loss_mean = val_loss_sum / N
        val_focal_pmean = val_focal_psum / N
        val_focal_nmean = val_focal_nsum / N
        self.logger.experiment.add_scalar('Loss/val', val_loss_mean.item(), self.current_epoch)
        self.log('val_loss', val_loss_mean)
        self.log('val_focal+', val_focal_pmean)
        self.log('val_focal-', val_focal_nmean)
        print('val_focal+',val_focal_pmean, 'val_focal-',val_focal_nmean )
        #d_stats = self._D_stats.get()
        #self.log('D_auc', d_stats['auc'])
        #self.log('D_pos_acc', d_stats['pos_accuracy'])
        #self.log('D_neg_acc', d_stats['neg_accuracy'])

        #pdsr = self._val_criterion.get()['Dotmetric']
        #self._val_criterion.reset()
        #self.log('pdsr', pdsr)

    def on_epoch_end(self, *args):
        self._G_loss.reset()
        #self._GD_loss.reset()
        self._recon_loss.reset()
        self._focal_p.reset()
        self._focal_n.reset()
        #self._D_loss.reset()
        #self._train_D_stats.reset()
        #self._D_stats.reset()
    

    def forward(self, input):
        output = self.G(input)
        if self._maxpool_atlast:           
          output = nn.MaxPool2d(6)(output) #在最後全部都做完後做一個MAXPOOL
        return output

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            dirpath=self._ckp_dir,
            filename=self._ckp_prefix+'_{epoch}_{val_loss:.6f}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min')