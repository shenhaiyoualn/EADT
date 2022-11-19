from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from config.option import TrainOptions
from model.total_model.smodel import *
opt = TrainOptions().parse()


def train():
    model = parsing(opt)
    save_path = "{}/{}/{}/{}/{}".format(opt.checkpoints_dir, opt.name, opt.ver, opt.dataset_name, opt.log_name)
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        save_top_k=3,
        save_weights_only=True,
        monitor='fid',
        mode='min',
        verbose=True
    )
    logger_path = "{}/{}/{}".format(opt.log_dir, opt.name, opt.dataset_name)
    logger = TensorBoardLogger(
        save_dir=logger_path,
        name=opt.log_name,
        version=opt.ver
    )
    if opt.use_amp:
        amp_level = 'O2'
    else:
        amp_level = 'O0'
    trainer = Trainer(
        fast_dev_run=opt.debug,
        logger=logger,
        max_epochs=opt.train_epoch,
        min_epochs=opt.train_epoch,
        checkpoint_callback=checkpoint,
        gpus=opt.gpu,
        check_val_every_n_epoch=opt.every_val_epoch,
        log_save_interval=2000,
        use_amp=opt.use_amp,
        amp_level=amp_level
    )
    trainer.fit(model)


train()
