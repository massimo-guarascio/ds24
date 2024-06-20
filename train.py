import argparse
import collections
import torch
import importlib
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 7563
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    trainer_name = config['trainer_name']
    module = importlib.import_module('trainer')
    trainer_class = getattr(module, trainer_name)
    print(f'Loaded trainer {trainer_class.__name__}')

    # setup data_loader instances
    dataset1_train = config.init_obj('dataset1_train', module_data)
    dataset1_train_loader = config.init_obj('dataset1_train_loader', module_data, dataset=dataset1_train)
    dataset1_adapt = config.init_obj('dataset1_adapt', module_data)
    dataset1_adapt_loader = config.init_obj('dataset1_adapt_loader', module_data, dataset=dataset1_adapt)

    dataset2_train = config.init_obj('dataset2_train', module_data)
    dataset2_train_loader = config.init_obj('dataset2_train_loader', module_data, dataset=dataset2_train)
    dataset2_adapt = config.init_obj('dataset2_adapt', module_data)
    dataset2_adapt_loader = config.init_obj('dataset2_adapt_loader', module_data, dataset=dataset2_adapt)

    dataset3_train = config.init_obj('dataset3_train', module_data)
    dataset3_train_loader = config.init_obj('dataset3_train_loader', module_data, dataset=dataset3_train)
    dataset3_adapt = config.init_obj('dataset3_adapt', module_data)
    dataset3_adapt_loader = config.init_obj('dataset3_adapt_loader', module_data, dataset=dataset3_adapt)

    dataset4_train = config.init_obj('dataset4_train', module_data)
    dataset4_train_loader = config.init_obj('dataset4_train_loader', module_data, dataset=dataset4_train)
    dataset4_adapt = config.init_obj('dataset4_adapt', module_data)
    dataset4_adapt_loader = config.init_obj('dataset4_adapt_loader', module_data, dataset=dataset4_adapt)

    dataset5_train = config.init_obj('dataset5_train', module_data)
    dataset5_train_loader = config.init_obj('dataset5_train_loader', module_data, dataset=dataset5_train)
    dataset5_adapt = config.init_obj('dataset5_adapt', module_data)
    dataset5_adapt_loader = config.init_obj('dataset5_adapt_loader', module_data, dataset=dataset5_adapt)


    # build model architecture, then print to console
    if 'arch' in config:
        model = config.init_obj('arch', module_arch)
        logger.info(model)
    else:
        model = None

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    if model:
        model = model.to(device)
    if len(device_ids) > 1 and model:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for disabling scheduler
    # if model:
    #     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #     optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    #     lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    # else:
    #     optimizer = None
    #     lr_scheduler = None

    dataset_list = [
        (dataset1_train_loader, dataset1_adapt_loader),
        (dataset2_train_loader, dataset2_adapt_loader),
        (dataset3_train_loader, dataset3_adapt_loader),
        (dataset4_train_loader, dataset4_adapt_loader),
        (dataset5_train_loader, dataset5_adapt_loader)
    ]

    trainer = trainer_class(metric_ftns=metrics, config=config, device=device,
                            dataset_list=dataset_list)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--sp', '--sample_perc'], type=float, target='sample_perc')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
