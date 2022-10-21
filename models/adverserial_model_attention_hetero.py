import torch, os
import torch.nn as nn
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from core.running_average import RunningAverage
from models.discriminator import Discriminator
from models.loss import get_criterion
from utils.analysis_utils import DiscriminatorStats
from utils.performance_diagram import PerformanceDiagramStable

from models.adverserial_model_PONI import BalAdvPoniModel

class TargetAttentionLean_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, prediction, add_t):
        p1 = prediction[:,None]
        input = torch.sigmoid(input)
        p1 = p1 * self.atten(input)
        return p1[:,0]
#     def forward(self, input , prediction , add_t):
#         p1 = prediction[:, None]
#         input = torch.sigmoid(input)
#         input = torch.concat((input, add_t), dim=1)
#         p1 = p1 * self.atten(input)
#         return p1[:,0], self.atten(input)
        
class TargetAttentionLean_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten = nn.Sequential(
            nn.Conv2d(2, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, prediction):
        p1 = prediction[:,None]
        input = torch.sigmoid(input)
        p1 = p1 * self.atten(input)
        return p1[:,0]
    
class TargetAttentionLean_3(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten1 = nn.Sequential(
            nn.Conv2d(2, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input, prediction):
        p1 = prediction[:,None]
        input = torch.sigmoid(input)
        p1 = p1 * self.atten1(input)
        return p1[:,0]
    
class BalAdvAttenHeteroModel(BalAdvPoniModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        train_data, train_label, train_mask, train_era5, train_dt = batch
        N = train_data.shape[0]
        if optimizer_idx == 0:
            loss_dict = self.generator_loss(train_data, train_label, train_mask, train_era5, train_dt)
            self._recon_loss.add(loss_dict['progress_bar']['recon_loss'].item() * N, N)
            self._GD_loss.add(loss_dict['progress_bar']['adv_loss'].item() * N, N)
            self._G_loss.add(loss_dict['loss'].item() * N, N)
            self.log('G', self._G_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        # train discriminator
        if optimizer_idx == 1:
            loss_dict = self.discriminator_loss(train_data, train_label, train_era5, train_dt)
            self._D_loss.add(loss_dict['loss'].item() * N, N)
            self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            # balance discriminator
            return loss_dict['loss'] * self._adv_w
        return loss_dict['loss']

    def generator_loss(self, input_data, target_label, target_mask, era_data, datetime):
        predicted_reconstruction = self(input_data, target_label, era_data, datetime)
        recons_loss = self._criterion(predicted_reconstruction, target_label, target_mask)
        # train generator
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)

        # adversarial loss is binary cross-entropy
        # ReLU is used since generator fools discriminator with -ve values
        adv_loss = self.adversarial_loss_fn(
            self.D(nn.ReLU()(predicted_reconstruction)).view(N, 1), valid, smoothing=False)
        tqdm_dict = {'recon_loss': recons_loss, 'adv_loss': adv_loss}
        loss = adv_loss * self._adv_w + recons_loss * (1 - self._adv_w)
        output = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'prediction': predicted_reconstruction})
        return output

    def discriminator_loss(self, input_data, target_label, era_data, datetime):
        predicted_reconstruction = self(input_data, target_label, era_data, datetime)
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)
        d_out = self.D(target_label)
        real_loss = self.adversarial_loss_fn(d_out.view(N, 1), valid)

        # how well can it label as fake?
        fake = torch.zeros(N, 1)
        fake = fake.type_as(input_data)
        # ReLU is used since generator fools discriminator with -ve values
        fake_loss = self.adversarial_loss_fn(self.D(nn.ReLU()(predicted_reconstruction.detach())).view(N, 1), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict})
        return output

    def forward(self, input, label, era5, datetime):
        # input: [B, 6, 2, H, W]; label: [B, 3, H, W]
        output_from_ecd = self.encoder(input) # output: [3][B, n_out, H', W']
        y_0 = torch.mean(input[:,:,-1], dim=1, keepdim=True) # y_0: [B, 1, H, W]
        tmp = [label[:, x:x+1] for x in range(0, label.size(1) - 1)]
        y = torch.cat((y_0, *tmp), dim=1) # [B, 3, H, W]
        del tmp
        pred = self.forecaster(list(output_from_ecd), y, self.aux,) # [3, B, H, W]
        output = torch.clone(pred)
        output[0] = self.attention1(input[:, -1:, 1], pred[0], datetime.squeeze()) # input shape: [batch, Seq, C, H, W]
        return output
    
    def configure_optimizers(self):
            opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                                 list(self.forecaster.parameters()) +
                                 list(self.aux1.parameters()) + 
                                 list(self.attention1.parameters()),
                                 lr=self.lr
                                 )
            opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
            return opt_g, opt_d


    def validation_step(self, batch, batch_idx):
        val_data, val_label, val_mask, val_era5, val_dt = batch
        # generator
        loss_dict = self.generator_loss(val_data, val_label, val_mask, val_era5, val_dt)
        aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        self._val_criterion.compute(aligned_prediction, val_label)

        # Discriminator stats
        pos = self.D(val_label)
        neg = self.D(aligned_prediction)
        self._D_stats.update(neg, pos)

        log_data = loss_dict.pop('progress_bar')
        return {'val_loss': log_data['recon_loss'], 'N': val_label.shape[0]}