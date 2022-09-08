from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import os

DATA_PATH = 'training/ocr/vietocr_data/'
CKPTS = 'training/ocr/checkpoint/seq2seq_finetuned_checkpoint.pth'
WEIGHTS = 'training/ocr/weights/seq2seq_finetuned.pth'
config = Cfg.load_config_from_name('vgg_seq2seq')

if not os.path.exists('/'.join(CKPTS.split('/')[:-1])):
    os.mkdir('/'.join(CKPTS.split('/')[:-1]))
if not os.path.exists('/'.join(WEIGHTS.split('/')[:-1])):
    os.mkdir('/'.join(WEIGHTS.split('/')[:-1]))

dataset_params = {
    'name':'vietocr',
    'data_root': DATA_PATH,
    'train_annotation':'train_annotations.txt',
    'valid_annotation':'valid_annotations.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':20000,
          'checkpoint': CKPTS,    
          'export':WEIGHTS,
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

trainer = Trainer(config, pretrained=True)
trainer.config.save('training/ocr/my_config.yml')
trainer.train()
