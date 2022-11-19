"""
@Author : Jeffrey Huang -> uryuohamusta@gmail.com
using torch-lighting model to train
"""
import random
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.kornia_d import P2Sset
from evaluation.FID import calc_fid_by_metrix
from model.discriminator import *
from model.generator.U_base import SPADEUNet, SPADEUNet_YPar
from model.loss.ganloss import GANLoss_ver2, hinge_panalty
from model.loss.gloss import GMSDLoss
from tools.scheduler import TTUR_GANLR
from utils.summary_util import loggger_group_image, save_tensor_to_disk, tensor_to_image
from utils.image_tool import par_tensor2pix
from torch import nn
import cv2
from model.generator.upsample import PixelShufflePack
from model.generator.pnet import SResBlock
import torch.nn.functional as F
class SAND_pix_BaseLine(pl.LightningModule):
    """
        This is the first baseline model:
            no vgg loss
            pixal loss L1
            use hinge loss
            only one scale D
    """

    def __init__(self, opt):
        super(SAND_pix_BaseLine, self).__init__()
        # some config
        opt.use_style_label = False
        self.opt = opt
        self.img_nc = opt.image_nc
        self.parssing_nc = opt.parsing_nc
        self.out_nc = opt.output_nc
        self.Generator = SPADEUNet(opt, in_channels=self.img_nc + self.parssing_nc, out_channels=self.out_nc)
        self.Discrimanator = MultiscaleDiscriminator(opt,
                                                     in_c=self.parssing_nc + self.out_nc)  # 19+1 img+parsing+//generated
        self.criterionGAN = GANLoss_ver2(
            opt.gan_mode, tensor=torch.FloatTensor, opt=opt)
        self.criterionPix = torch.nn.L1Loss()
        self.criterionPixy = torch.nn.L1Loss()
        self.train_set = P2Sset(self.opt)
        self.test_set = P2Sset(self.opt, forTrain=False)

    # dataloaders
    @pl.data_loader
    def train_dataloader(self):  # trainset
        dataset = self.train_set
        loader = DataLoader(dataset=dataset, shuffle=True, num_workers=8, batch_size=self.opt.bsize, pin_memory=True)
        return loader

    @pl.data_loader  # testset
    def val_dataloader(self):
        dataset = self.test_set
        loader = DataLoader(dataset=dataset, batch_size=self.opt.bsize, pin_memory=True)
        return loader

    # optimazor
    def configure_optimizers(self):
        generator_opt = Adam(self.Generator.parameters(), lr=self.opt.g_lr, betas=(self.opt.beta1, self.opt.beta2))
        disriminator_opt = Adam(self.Discrimanator.parameters(), lr=self.opt.d_lr,
                                betas=(self.opt.beta1, self.opt.beta2))
        # LR scheme
        if self.opt.no_TTUR:
            g_lrscd = TTUR_GANLR(generator_opt, self.opt.niter, self.opt.niter_decay, 'g')
            d_lrscd = TTUR_GANLR(disriminator_opt, self.opt.niter, self.opt.niter_decay, 'd')
            return [generator_opt, disriminator_opt], [g_lrscd, d_lrscd]
        else:
            return [generator_opt, disriminator_opt]

    # testing
    def forward(self, x, parsing):
        out = self.Generator.forward(x, parsing)
        return out

    # trainning
    def training_step(self, batch, batch_num, optimizer_idx):

        logger = self.logger.experiment
        result = {}
        if optimizer_idx is 0:  # g optimazor
            x, y = self.train_set.apply_tranform(batch)
            if self.opt.use_amp:
                x = x.half()
                y = y.half()
            parsing = x[:, 3:]
            input = x
            y_parsing = y[:, 1:]
            y = y[:, :1]
            x = x[:, :3]
            # 1.442 加入的属性
            if self.opt.gamma_beta_mode in ['final', 'feature']:
                fake_image, x_gamma_beta = self.Generator.forward(input, parsing, gamma_mode='final')
            else:
                fake_image = self.Generator.forward(input, parsing)

            pred_fake, pred_real = self.discriminate(parsing, fake_image, y)
            gen_g_loss = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan

            gen_pix_loss = self.criterionPix(fake_image, y) * self.opt.lambda_pix
            g_loss = gen_g_loss + gen_pix_loss

            if self.global_step % 100 is 0:
                loggger_group_image(logger=logger, step=self.global_step, fake_img=fake_image, org_img=x, target_img=y,
                                    tag="train")
            result = {
                "loss": g_loss,
                "progress_bar": {
                    "g_loss_gen": gen_g_loss,

                },
                "log": {
                    "g_loss_gen": gen_g_loss,
                    "g_loss_total": g_loss
                }
            }
            result["progress_bar"]["g_loss_pix"] = gen_pix_loss
            result["log"]["g_loss_pix"] = gen_pix_loss


        elif optimizer_idx is 1:  # d optimazer
            x, y = self.train_set.apply_tranform(batch)
            if self.opt.use_amp:
                x = x.half()
                y = y.half()
            parsing = x[:, 3:]
            input = x
            y = y[:, :1]
            with torch.no_grad():
                fake_image = self.Generator.forward(input, parsing)
                fake_image = fake_image.detach()
            pred_fake, pred_real = self.discriminate(parsing, fake_image, y)
            d_fake_loss = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.lambda_gan
            d_real_loss = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.lambda_gan
            d_loss = d_fake_loss + d_real_loss
            result = {
                "loss": d_loss,
                "progress_bar": {
                    "d_loss_total": d_loss,
                },
                "log": {
                    "d_loss_total": d_loss,
                    "d_loss_fake": d_fake_loss,
                    "d_loss_real": d_real_loss,

                }
            }
        return result

    def validation_step(self, batch, batch_num):
        x, y = self.test_set.apply_tranform(batch)
        y = y[:, :1]

        if self.opt.use_amp:
            x = x.half()
            y = y.half()
        x_parsing = x[:, 3:]

        input = x
        x = x[:, :3]
        fake_image = self.Generator.forward(input, x_parsing, x_parsing)

        if self.opt.use_amp:
            x = x.float()
            y = y.float()
            fake_image = fake_image.float()
        result = {
            'org_img': x.cpu(),
            'gen_img': fake_image.detach().cpu(),
            'tar_img': y.cpu()
        }
        return result

    def validation_end(self, outputs):
        org_img_all = []
        gen_img_all = []
        y_img_all = []

        for elem in outputs:
            org_img = elem['org_img']
            gen_img = elem['gen_img']
            y_img = elem['tar_img']
            paddingSize1, paddingSize2 = int((self.opt.input_size - self.opt.img_h) / 2), int(
                (self.opt.input_size - self.opt.img_w) / 2)
            org_img_all.append(org_img[:, :, paddingSize1:self.opt.input_size - paddingSize1,
                               paddingSize2:self.opt.input_size - paddingSize2].detach())
            gen_img_all.append(gen_img[:, :, paddingSize1:self.opt.input_size - paddingSize1,
                               paddingSize2:self.opt.input_size - paddingSize2].detach())
            y_img_all.append(y_img[:, :, paddingSize1:self.opt.input_size - paddingSize1,
                             paddingSize2:self.opt.input_size - paddingSize2].detach())

        org_img_all = torch.cat(org_img_all, dim=0)
        gen_img_all = torch.cat(gen_img_all, dim=0)
        y_img_all = torch.cat(y_img_all, dim=0)
        # expand tensor sketch to 3 channal
        gen_img_all = gen_img_all.expand_as(org_img_all)
        y_img_all = y_img_all.expand_as(org_img_all)
        # 注意此时还是-1~1
        # fid 需要0~1
        fid_gen = tensor_to_image(gen_img_all)
        fid_y = tensor_to_image(y_img_all)
        fid = calc_fid_by_metrix(fid_gen, fid_y, device="cpu", bsize=10)  # use cpu to clac fid
        torch.cuda.empty_cache()
        # logger
        logger = self.logger.experiment
        img_len = gen_img_all.shape[0]
        sample_ind = random.sample(range(img_len), min(20, img_len))

        loggger_group_image(logger=logger, step=self.global_step, fake_img=gen_img_all[sample_ind],
                            org_img=org_img_all[sample_ind],
                            target_img=y_img_all[sample_ind], tag="val")
        # 保存部分

        save_path = "{}/{}/{}/{}/{}/{}".format(self.opt.result_img_dir, self.opt.name, self.opt.ver,
                                               self.opt.log_name,
                                               self.opt.dataset_name, self.current_epoch)
        save_tensor_to_disk(path=save_path, img_list=[org_img_all, y_img_all, gen_img_all])

        return {
            'progress_bar': {'val_fid': fid},
            'log': {'fid': fid},
            'fid': fid
        }

    # tools
    def __clac_dloss__(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        GAN_Feat_loss = torch.zeros([1], dtype=torch.float).to("cuda:{}".format(self.opt.gpu))
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss

    def discriminate(self, input_semantics, fake_image, real_image):
        # torch.nn.functional.upsample(input_semantics, scale_factor=2, mode='bicubic')  这两行好像没用
        # torch.nn.functional.upsample(real_image, scale_factor=2, mode='bicubic')

        # F.pixel_shuffle(inputs, upscale_factor)

        # F.interpolate(input_semantics,2,mode="bilinear")
        # F.interpolate(real_image,2,mode="bilinear")
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.Discrimanator(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def discriminate_with_X(self, input_semantics, org_img, fake_image, real_image):
        fake_concat = torch.cat([org_img, input_semantics, fake_image], dim=1)
        real_concat = torch.cat([org_img, input_semantics, real_image], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.Discrimanator(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real


class parsing(SAND_pix_BaseLine):
    def __init__(self, opt):
        opt.spade_mode = 'res2'
        opt.total_label = 0
        opt.gamma_beta_criterion = False
        super(parsing, self).__init__(opt)
        self.Generator = SPADEUNet_YPar(opt, self.img_nc, self.parssing_nc, self.out_nc)
        self.criterionPar = torch.nn.MSELoss()
        if self.opt.use_gmsd:
            self.criterionGMSD = GMSDLoss(self.img_nc)

    def configure_optimizers(self):
        generator_opt = Adam(self.Generator.parameters(), lr=self.opt.g_lr, betas=(self.opt.beta1, self.opt.beta2))
        disriminator_opt = Adam(self.Discrimanator.parameters(), lr=self.opt.d_lr,
                                betas=(self.opt.beta1, self.opt.beta2))
        opt_list = [generator_opt, disriminator_opt]

        # LR scheme
        lr_list = []
        g_lrscd = TTUR_GANLR(generator_opt, self.opt.niter, self.opt.niter_decay, 'g')
        d_lrscd = TTUR_GANLR(disriminator_opt, self.opt.niter, self.opt.niter_decay, 'd')
        lr_list.append(g_lrscd)
        lr_list.append(d_lrscd)

        return opt_list, lr_list

    def forward(self, x, parsing):
        img, pars = self.Generator.forward(x, parsing)
        return img, pars

    def training_step(self, batch, batch_num, optimizer_idx):
        logger = self.logger.experiment
        result = {}
        if optimizer_idx is 0:  # g optimazor
            x, y = self.train_set.apply_tranform(batch)
            if self.opt.use_amp:
                x = x.half()
                y = y.half()
            x_parsing = x[:, self.img_nc:]
            x = x[:, :self.img_nc]
            y_parsing = y[:, self.out_nc:]
            y = y[:, :self.out_nc]
            if self.opt.x_parsing:
                fake_image, fake_par = self.Generator.forward(x, x_parsing, x_parsing)
                pred_fake, pred_real = self.discriminate(x_parsing, fake_image, y)
            else:
                fake_image, fake_par = self.Generator.forward(x, x_parsing, y_parsing)
                pred_fake, pred_real = self.discriminate(y_parsing, fake_image, y)

            gen_g_loss = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
            gen_pix_loss = self.criterionPix(fake_image, y) * self.opt.lambda_pix
            if self.opt.x_parsing:
                gen_par_loss = torch.sqrt(self.criterionPar(fake_par, x_parsing)) * self.opt.lambda_par
            else:
                gen_par_loss = torch.sqrt(self.criterionPar(fake_par, y_parsing)) * self.opt.lambda_par
            g_loss = gen_g_loss + gen_pix_loss + gen_par_loss
            # GMSD Loss
            if self.opt.use_gmsd:
                gmsd_loss = self.criterionGMSD(torch.cat([fake_image, fake_image, fake_image], dim=1),
                                               x) * self.opt.lambda_gmsd
                g_loss = g_loss + gmsd_loss

            if self.global_step % 200 is 0:
                loggger_group_image(logger=logger, step=self.global_step, fake_img=fake_image, org_img=x, target_img=y,
                                    tag="train")
                # log par img
                par_fakeimg = par_tensor2pix(fake_par, self.opt.parsing_nc, one_hot=False)
                if self.opt.x_parsing:
                    par_real = par_tensor2pix(x_parsing, self.opt.parsing_nc, one_hot=False)
                else:
                    par_real = par_tensor2pix(y_parsing, self.opt.parsing_nc, one_hot=False)
                loggger_group_image(logger=logger, step=self.global_step, fake_img=par_fakeimg, org_img=x,
                                    target_img=par_real, tag="train_par")
            result = {
                "loss": g_loss,
                "progress_bar": {
                    "g_loss_gen": gen_g_loss,
                    "g_loss_other": gen_pix_loss + gen_par_loss,
                },
                "log": {
                    "g_loss_gen": gen_g_loss,
                    "g_loss_pix": gen_pix_loss,
                    "g_loss_par": gen_par_loss,
                    "g_loss_total": gen_pix_loss + gen_g_loss + gen_par_loss
                }
            }
            if self.opt.use_gmsd:
                result["log"]["g_loss_gmsd"] = gmsd_loss
                result["progress_bar"]["g_loss_other"] += gmsd_loss


        elif optimizer_idx is 1:  # d optimazer
            x, y = self.train_set.apply_tranform(batch)
            if self.opt.use_amp:
                x = x.half()
                y = y.half()
            x_parsing = x[:, self.img_nc:]
            x = x[:, :self.img_nc]
            y_parsing = y[:, self.out_nc:]
            y = y[:, :self.out_nc]
            with torch.no_grad():
                if self.opt.x_parsing:
                    fake_image, fake_par = self.Generator.forward(x, x_parsing, x_parsing)
                else:
                    fake_image, fake_par = self.Generator.forward(x, x_parsing, y_parsing)
                fake_image = fake_image.detach()
            if self.opt.x_parsing:
                pred_fake, pred_real = self.discriminate(x_parsing, fake_image, y)
            else:
                pred_fake, pred_real = self.discriminate(y_parsing, fake_image, y)
            d_fake_loss = self.criterionGAN(pred_fake, False,
                                            for_discriminator=True) * self.opt.lambda_gan
            d_real_loss = self.criterionGAN(pred_real, True,
                                            for_discriminator=True) * self.opt.lambda_gan
            if self.opt.gan_mode == "hinge":
                d_panalty_loss = hinge_panalty(pred_real, pred_fake) * self.opt.lambda_panalty
                d_loss = d_fake_loss + d_real_loss + d_panalty_loss
            else:
                d_loss = d_fake_loss + d_real_loss
            result = {
                "loss": d_loss,
                "progress_bar": {
                    "d_loss_total": d_loss,
                },
                "log": {
                    "d_loss_total": d_loss,
                    "d_loss_fake": d_fake_loss,
                    "d_loss_real": d_real_loss,

                }
            }
        return result

    def validation_step(self, batch, batch_num):
        x, y = self.test_set.apply_tranform(batch)
        if self.opt.use_amp:
            x = x.half()
            y = y.half()
        x_parsing = x[:, self.img_nc:]
        y_parsing = y[:, self.out_nc:]
        x = x[:, :self.img_nc]
        y = y[:, :self.out_nc]
        fake_image_x, fake_par_x = self.Generator.forward(x, x_parsing, x_parsing)
        fake_image_y, fake_par_y = self.Generator.forward(x, x_parsing, y_parsing)
        fake_par_x_img = par_tensor2pix(fake_par_x, self.opt.parsing_nc, False)
        fake_par_y_img = par_tensor2pix(fake_par_y, self.opt.parsing_nc, False)
        if self.opt.x_parsing:
            real_par_img = par_tensor2pix(x_parsing, self.opt.parsing_nc, False)
        else:
            real_par_img = par_tensor2pix(y_parsing, self.opt.parsing_nc, False)
        if self.opt.use_amp:
            x = x.float()
            y = y.float()
            fake_image_x = fake_image_x.float()
            fake_image_y = fake_image_y.float()

        result = {
            'org_img': x.cpu(),
            'gen_img_x': fake_image_x.detach().cpu(),
            'gen_img_y': fake_image_y.detach().cpu(),
            'tar_img': y.cpu(),
            'par_fake_x_img': fake_par_x_img.cpu(),
            'par_fake_y_img': fake_par_y_img.cpu(),
            'par_real_img': real_par_img.cpu()

        }
        return result

    def validation_end(self, outputs):
        org_img_all = []
        gen_img_x_all = []
        gen_img_y_all = []
        y_img_all = []
        fake_par_x_all = []
        fake_par_y_all = []
        real_par_all = []
        pad_h = (self.opt.input_size - self.opt.img_h) // 2
        pad_w = (self.opt.input_size - self.opt.img_w) // 2
        for elem in outputs:
            org_img = elem['org_img']
            org_img_all.append(
                org_img[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            gen_x_img = elem['gen_img_x']
            gen_img_x_all.append(
                gen_x_img[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            gen_y_img = elem['gen_img_y']
            gen_img_y_all.append(
                gen_y_img[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            y_img = elem['tar_img']
            y_img_all.append(y_img[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            fake_par_x = elem['par_fake_x_img']
            fake_par_x_all.append(
                fake_par_x[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            fake_par_y = elem['par_fake_y_img']
            fake_par_y_all.append(
                fake_par_y[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())
            real_par = elem['par_real_img']
            real_par_all.append(
                real_par[:, :, pad_h:self.opt.input_size - pad_h, pad_w:self.opt.input_size - pad_w].detach())

        org_img_all = torch.cat(org_img_all, dim=0)
        gen_img_x_all = torch.cat(gen_img_x_all, dim=0)
        gen_img_y_all = torch.cat(gen_img_y_all, dim=0)
        y_img_all = torch.cat(y_img_all, dim=0)
        fake_par_x_all = torch.cat(fake_par_x_all, dim=0)
        fake_par_y_all = torch.cat(fake_par_y_all, dim=0)
        real_par_all = torch.cat(real_par_all, dim=0)
        # expand tensor sketch to 3 channal
        gen_img_x_all = gen_img_x_all.expand_as(org_img_all)
        gen_img_y_all = gen_img_y_all.expand_as(org_img_all)
        y_img_all = y_img_all.expand_as(org_img_all)
        # 注意此时还是-1~1
        # fid 需要0~1
        fid_gen_y = tensor_to_image(gen_img_y_all)
        fid_gen_x = tensor_to_image(gen_img_x_all)
        fid_y_img = tensor_to_image(y_img_all)
        fid_y = calc_fid_by_metrix(fid_gen_y, fid_y_img, device="cpu", bsize=10)  # use cpu to clac fid
        fid_x = calc_fid_by_metrix(fid_gen_x, fid_y_img, device="cpu", bsize=10)  # use cpu to clac fid
        torch.cuda.empty_cache()
        # logger
        logger = self.logger.experiment
        img_len = gen_img_y_all.shape[0]
        sample_ind = random.sample(range(img_len), min(20, img_len))

        # 保存部分

        save_path = "{}/{}/{}/{}/{}/{}".format(self.opt.result_img_dir, self.opt.name, self.opt.ver,
                                               self.opt.log_name,
                                               self.opt.dataset_name, self.current_epoch)
        save_tensor_to_disk(path=save_path + "/x",
                            img_list=[org_img_all, y_img_all, gen_img_x_all, real_par_all, fake_par_x_all])
        save_tensor_to_disk(path=save_path + "/y",
                            img_list=[org_img_all, y_img_all, gen_img_y_all, real_par_all, fake_par_y_all])
        if self.opt.x_parsing:
            loggger_group_image(logger=logger, step=self.global_step, fake_img=gen_img_x_all[sample_ind],
                                org_img=org_img_all[sample_ind],
                                target_img=y_img_all[sample_ind], tag="val")

            loggger_group_image(logger=logger, step=self.global_step, fake_img=fake_par_x_all[sample_ind],
                                org_img=org_img_all[sample_ind],
                                target_img=real_par_all[sample_ind], tag="val_par")
            return {
                'progress_bar': {'val_fid_x': fid_x, 'val_fid_y': fid_y},
                'log': {'fid_x': fid_x, 'fid_y': fid_y},
                'fid': fid_x
            }
        else:
            loggger_group_image(logger=logger, step=self.global_step, fake_img=gen_img_y_all[sample_ind],
                                org_img=org_img_all[sample_ind],
                                target_img=y_img_all[sample_ind], tag="val")

            loggger_group_image(logger=logger, step=self.global_step, fake_img=fake_par_y_all[sample_ind],
                                org_img=org_img_all[sample_ind],
                                target_img=real_par_all[sample_ind], tag="val_par")
            return {
                'progress_bar': {'val_fid_x': fid_x, 'val_fid_y': fid_y},
                'log': {'fid_x': fid_x, 'fid_y': fid_y},
                'fid': fid_y
            }
