import re
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.data_loaders import TextDataset, create_fast_dataloader_from_dataset, \
    TextDatasetWithSource
from logger import TensorboardWriter
from model.loss import BinaryCrossEntropy, MoELoss, LossResult, MoEBCELoss, MoELossWithDomain, \
    LossResultWithDomain
from model.metric import get_label_classification_report
from model.model import BaseClassifier, MoE, MaxProbabilityModel, MoESparseModel, \
    MoEWithDomainDetector
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import MetricTracker, save_results_to_file
from sklearn.metrics import classification_report as classification_report_sklearn


def save_model(model, f_name):
    print('save model to', f_name)
    current_device = next(model.parameters()).device
    torch.save(model.cpu().state_dict(), f_name)
    model.to(current_device)

def reset_weights(m):
  """
    Try resetting model weights to avoid
    weight leakage.
  """
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()

class TrainerCrossDomainMoeBaseModel:
    """
    Trainer class for cross-domain moe architecture
    """
    def __init__(self, metric_ftns, config, device,
                 dataset_list, **kwargs):
        """

        :param metric_ftns: list of metric functions
        :param config: config dict
        :param device: torch device
        :param dataset_list: list of pairs (datasetX_train_loader, datasetX_adapt_loader)
        :param kwargs:
        """
        assert dataset_list and len(dataset_list) == 5
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        self.dataset1_train, self.dataset1_adapt = dataset_list[0]
        self.dataset2_train, self.dataset2_adapt = dataset_list[1]
        self.dataset3_train, self.dataset3_adapt = dataset_list[2]
        self.dataset4_train, self.dataset4_adapt = dataset_list[3]
        self.dataset5_train, self.dataset5_adapt = dataset_list[4]

        self.log_step = int(np.sqrt(self.dataset1_train.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'train_{c}/loss' for c in 'ABCDE']
        m += [f'train_{c}/loss_epoch' for c in 'ABCDE']
        m += [f'train_{c}/metrics/{m}' for c in 'ABCDE' for m in get_label_classification_report()]

        m += [f'adapt_{i}_{j}/loss' for i in ['A', 'B', 'C', 'D', 'E']
              for j in ['A', 'B', 'C', 'D', 'E']  if i!=j]
        m += [f'adapt_{i}_{j}/loss_epoch' for i in ['A', 'B', 'C', 'D', 'E']
              for j in ['A', 'B', 'C', 'D', 'E']  if i!=j]
        m += [f'adapt_{i}_{j}/metrics/{m}' for i in ['A', 'B', 'C', 'D', 'E']
              for j in ['A', 'B', 'C', 'D', 'E']  if i!=j for m in get_label_classification_report()]
        self.train_metrics = MetricTracker(*m, writer=self.writer)

        m = [f'valid_{c}/loss' for c in 'ABCDE']
        m += [f'valid_{c}/loss_epoch' for c in 'ABCDE']
        m += [f'valid_{c}/metrics/{m}' for c in 'ABCDE' for m in get_label_classification_report()]
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0
        batch_size_1 = self.config['dataset1_train_loader/args/batch_size'] or 16
        batch_size_2 = self.config['dataset2_train_loader/args/batch_size'] or 16
        batch_size_3 = self.config['dataset3_train_loader/args/batch_size'] or 16
        batch_size_4 = self.config['dataset4_train_loader/args/batch_size'] or 16
        batch_size_5 = self.config['dataset5_train_loader/args/batch_size'] or 16
        batch_size_fine = self.config['dataset1_adapt_loader/args/batch_size']

        loss_function = BinaryCrossEntropy()
        modelA = self.train_on_dataset(self.dataset1_train, self.dataset1_adapt, batch_size_1, loss_function,
                                       log_prefix='A')
        self.adapt_on_dataset(self.dataset2_adapt, batch_size_fine, modelA, loss_function,
                              log_prefix='A_B')
        self.adapt_on_dataset(self.dataset3_adapt, batch_size_fine, modelA, loss_function,
                                        log_prefix='A_C')
        self.adapt_on_dataset(self.dataset4_adapt, batch_size_fine, modelA, loss_function,
                                        log_prefix='A_D')
        self.adapt_on_dataset(self.dataset5_adapt, batch_size_fine, modelA, loss_function,
                                        log_prefix='A_E')

        loss_function = BinaryCrossEntropy()
        modelB = self.train_on_dataset(self.dataset2_train, self.dataset2_adapt, batch_size_2, loss_function, log_prefix='B')
        self.adapt_on_dataset(self.dataset1_adapt, batch_size_fine, modelB, loss_function,
                                         log_prefix='B_A')
        self.adapt_on_dataset(self.dataset3_adapt, batch_size_fine, modelB, loss_function,
                                         log_prefix='B_C')
        self.adapt_on_dataset(self.dataset4_adapt, batch_size_fine, modelB, loss_function,
                                         log_prefix='B_D')
        self.adapt_on_dataset(self.dataset5_adapt, batch_size_fine, modelB, loss_function,
                                         log_prefix='B_E')

        loss_function = BinaryCrossEntropy()
        modelC = self.train_on_dataset(self.dataset3_train, self.dataset3_adapt, batch_size_3, loss_function, log_prefix='C')
        self.adapt_on_dataset(self.dataset1_adapt, batch_size_fine, modelC, loss_function,
                                         log_prefix='C_A')
        self.adapt_on_dataset(self.dataset2_adapt, batch_size_fine, modelC, loss_function,
                                         log_prefix='C_B')
        self.adapt_on_dataset(self.dataset4_adapt, batch_size_fine, modelC, loss_function,
                                         log_prefix='C_D')
        self.adapt_on_dataset(self.dataset5_adapt, batch_size_fine, modelC, loss_function,
                                         log_prefix='C_E')

        loss_function = BinaryCrossEntropy()
        modelD = self.train_on_dataset(self.dataset4_train, self.dataset4_adapt, batch_size_4, loss_function, log_prefix='D')
        self.adapt_on_dataset(self.dataset1_adapt, batch_size_fine, modelD, loss_function,
                                         log_prefix='D_A')
        self.adapt_on_dataset(self.dataset2_adapt, batch_size_fine, modelD, loss_function,
                                         log_prefix='D_B')
        self.adapt_on_dataset(self.dataset3_adapt, batch_size_fine, modelD, loss_function,
                                         log_prefix='D_C')
        self.adapt_on_dataset(self.dataset5_adapt, batch_size_fine, modelD, loss_function,
                                         log_prefix='D_E')

        loss_function = BinaryCrossEntropy()
        modelE = self.train_on_dataset(self.dataset5_train, self.dataset5_adapt, batch_size_5, loss_function, log_prefix='E')
        self.adapt_on_dataset(self.dataset1_adapt, batch_size_fine, modelE, loss_function,
                                         log_prefix='E_A')
        self.adapt_on_dataset(self.dataset2_adapt, batch_size_fine, modelE, loss_function,
                                         log_prefix='E_B')
        self.adapt_on_dataset(self.dataset3_adapt, batch_size_fine, modelE, loss_function,
                                        log_prefix='E_C')
        self.adapt_on_dataset(self.dataset4_adapt, batch_size_fine, modelE, loss_function,
                                         log_prefix='E_D')

    def train_on_dataset(self, dataset, valdataset, batch_size, loss_function, num_epochs=100, log_prefix='A'):
        best_loss = np.inf

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valdataset, batch_size=batch_size)
        len_epoch = len(trainloader)

        network = BaseClassifier(self.text_embedding_size).to(self.device)
        network.apply(reset_weights)

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        for epoch in range(0, num_epochs):
            current_loss = [0]
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                logit = network(inputs)
                loss = loss_function(F.sigmoid(logit), targets)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'train_{log_prefix}/loss', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss[0] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # val_loss = self.evaluate(loss_function, network, valloader, log_prefix=log_prefix)
            loss_batch = current_loss[0] / batch_number
            if loss_batch < best_loss:
                best_loss = loss_batch
                save_model(network, filename)

        self.logger.info('Training process has finished.')

        if Path(filename).exists():
            best_model = network.set_state_dict(torch.load(filename)).to(self.device)
        else:
            best_model = network

        return best_model

    def evaluate(self, loss_function, network, loader, log_prefix='A'):
        current_loss = 0.
        network.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logit = network(inputs)
                loss = loss_function(F.sigmoid(logit), targets)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'valid_{log_prefix}/loss', loss.item())

                current_loss += loss.item()

                output = F.sigmoid(logit)

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(targets.cpu().numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'valid_{log_prefix}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'valid_{log_prefix}'):
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        if 'metric_validation' in self.config:
            metric_name = self.config["metric_validation"]
            return log[metric_name] if metric_name in log else None
        return current_loss / len(loader)

    def adapt_on_dataset(self, dataset, batch_size, network, loss_function, num_epochs=100, log_prefix='A'):
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        len_epoch = len(trainloader)

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        for epoch in range(0, num_epochs):
            current_loss = [0]
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                logit = network(inputs)
                loss = loss_function(F.sigmoid(logit), targets)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'adapt_{log_prefix}/loss', loss.item())

                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train (adaptation) Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'adapt_{log_prefix}/loss_epoch', current_loss[0] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'adapt_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished.')

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)

        return network


class TrainerCrossDomainMoe:
    """
    Trainer class for cross-domain moe architecture
    """
    def __init__(self, metric_ftns, config, device,
                 dataset_list, **kwargs):
        """

        :param metric_ftns: list of metric functions
        :param config: config dict
        :param device: torch device
        :param dataset_list: list of pairs (datasetX_train_loader, datasetX_adapt_loader)
        :param kwargs:
        """
        assert dataset_list and len(dataset_list) > 0
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        dir_path = Path(self.config['expert_model_dump_dir'])
        self.expert_model_dump = [ dir_path / x for x in self.config['expert_model_dump']]
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        self.datasets = self._process_datasets(dataset_list)
        self.expert_model_dump_by_dataset = self._group_models_by_dataset()

        self.log_step = int(np.sqrt(self.config['dataset1_train_loader/args/batch_size'] or 16))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'train_{c}/{metric_name}' for c in 'ABCDE' for metric_name in LossResultWithDomain._fields]
        m += [f'train_{c}/loss_epoch' for c in 'ABCDE']

        self.train_metrics = MetricTracker(*m, writer=self.writer)

        m = [f'valid_{c}/{metric_name}' for c in 'ABCDE' for metric_name in LossResultWithDomain._fields]
        m += [f'valid_{c}/loss_epoch' for c in 'ABCDE']
        m += [f'valid_{c}/metrics/{m}' for c in 'ABCDE' for m in get_label_classification_report()]
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        self.training_step = 0

    def _process_datasets(self, dataset_list):
        return [pd.concat((a.dataset.data, b.dataset.data), axis=0)
                          for a,b in dataset_list]

    def _group_models_by_dataset(self):
        num_dataset = len(self.datasets)
        groups = {chr(ord('A') + i): [] for i in range(num_dataset)}
        for model in self.expert_model_dump:
            model_label = re.findall(r'model-([A-Z_]+).pth', model.name)[0]

            for k, v in groups.items():
                if k not in model_label:
                    v.append(model)

        return groups

    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        batch_sizes = [32] * len(batch_sizes)

        loss_function = MoELoss(
            lambda_l1_gate=self.config['loss/lambda_l1_gate'],
            lambda_l1_exp=0,
            lambda_balancing=self.config['loss/lambda_balancing'],
            class_weights=self.config['loss/class_weights'])

        num_workers = self.config['dataset1_train_loader/args/num_workers']

        for i, test_dataset in enumerate(self.datasets):
            train_dataset = [x for j, x in enumerate(self.datasets) if i != j]
            train_dataset = TextDataset(pd.concat(train_dataset, axis=0))
            test_dataset = TextDataset(pd.concat(test_dataset, axis=0))

            # train_data_loader = DataLoader(train_dataset, batch_size=batch_sizes[i],
            #                                shuffle=True, num_workers=num_workers)
            # test_data_loader = DataLoader(test_dataset, batch_size=batch_sizes[i],
            #                                shuffle=False, num_workers=num_workers)

            train_data_loader = create_fast_dataloader_from_dataset(train_dataset,
                                                                    batch_size=batch_sizes[i],
                                                                    shuffle=True)

            test_data_loader = create_fast_dataloader_from_dataset(test_dataset,
                                                                   batch_size=batch_sizes[i],
                                                                   shuffle=False)

            self.train_on_dataset(train_data_loader, test_data_loader,
                                  loss_function, log_prefix=chr(ord('A') + i))

        print(f'Sim end. Result folder is {self.config.save_dir}')
        output_result_file = self.config.save_dir / "results_macro_f1.xlsx"
        if output_result_file.exists():
            df = pd.read_excel(output_result_file)
            print(df.head(100))

    def train_on_dataset(self, train_loader, val_loader, loss_function, log_prefix='A'):
        best_loss = np.inf

        num_epochs = self.config['trainer/epochs']
        len_epoch = len(train_loader)

        network = MoE(
                      self.text_embedding_size,
                      self.expert_model_dump_by_dataset[log_prefix],
                      trainable_expert=self.config['trainable_expert'],
                      use_hard_prediction=self.config['use_hard_prediction'])
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        self.logger.info(f'Model:\n {network}')

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, network)

                loss.loss.backward()
                optimizer.step()

                current_loss += loss.loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                [self.train_metrics.update(f'train_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            val_loss = self.evaluate(loss_function, network, val_loader,
                                     log_prefix=log_prefix, epoch_id=epoch)
            if val_loss < best_loss:
                filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
                save_model(network, filename)
                best_loss = val_loss

        self.logger.info('Training process has finished.')

    def evaluate(self, loss_function, network, loader, log_prefix='A', epoch_id=0):
        current_loss = 0.
        network.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, network, validation=True)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                [self.valid_metrics.update(f'valid_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                current_loss += loss.loss.item()

                output = moe_outputs.weighted_outputs

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(targets.cpu().numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'valid_{log_prefix}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'valid_{log_prefix}'):
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        # save results to file
        desc_dict = {
            'DATASET': log_prefix,
            'run_name': self.config['name'],
            'run_id': self.config.save_dir.name,
            'epoch': epoch_id
        }
        output_file = self.config.save_dir / 'results.xlsx'
        save_results_to_file(output_file, log, desc_dict)

        if 'metric_validation' in self.config:
            metric_name = self.config["metric_validation"]
            return log[metric_name] if metric_name in log else None
        return current_loss / len(loader)


class TrainerCrossDomainMoeMaxProbability(TrainerCrossDomainMoe):
    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        loss_function = MoEBCELoss(class_weights=self.config['loss/class_weights'])

        num_workers = self.config['dataset1_train_loader/args/num_workers']

        for i, test_dataset in enumerate(self.datasets):
            # train_dataset = [x for j, x in enumerate(self.datasets) if i != j]
            # train_dataset = TextDataset(pd.concat(train_dataset, axis=0))

            test_dataset = TextDataset(test_dataset)

            # train_data_loader = DataLoader(train_dataset, batch_size=batch_sizes[i],
            #                                shuffle=True, num_workers=num_workers)
            # test_data_loader = DataLoader(test_dataset, batch_size=batch_sizes[i],
            #                                shuffle=False, num_workers=num_workers)

            test_data_loader = create_fast_dataloader_from_dataset(test_dataset,
                                                                   batch_size=batch_sizes[i],
                                                                   shuffle=False)

            log_prefix = chr(ord('A') + i)

            network = MaxProbabilityModel(self.text_embedding_size,
                          self.expert_model_dump_by_dataset[log_prefix])
            network.to(self.device)

            self.evaluate(loss_function, network, test_data_loader,
                          log_prefix=log_prefix, epoch_id=0)


class TrainerCrossDomainStacking(TrainerCrossDomainMoe):
    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        loss_function = MoEBCELoss(class_weights=self.config['loss/class_weights'])

        num_workers = self.config['dataset1_train_loader/args/num_workers']

        for i, test_dataset in enumerate(self.datasets):
            train_dataset = [x for j, x in enumerate(self.datasets) if i != j]
            train_dataset = TextDataset(pd.concat(train_dataset, axis=0))

            test_dataset = TextDataset(test_dataset)

            train_data_loader = DataLoader(train_dataset, batch_size=batch_sizes[i],
                                           shuffle=True, num_workers=num_workers)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_sizes[i],
                                           shuffle=False, num_workers=num_workers)

            self.train_on_dataset(train_data_loader, test_data_loader,
                                  loss_function, log_prefix=chr(ord('A') + i))

    def train_on_dataset(self, train_loader, val_loader, loss_function, log_prefix='A'):
        best_loss = np.inf

        num_epochs = self.config['trainer/epochs']
        len_epoch = len(train_loader)

        network = MoE(self.text_embedding_size,
                      self.expert_model_dump_by_dataset[log_prefix],
                      use_hard_prediction=self.config['use_hard_prediction'])
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, network)
                loss.loss.backward()
                optimizer.step()

                current_loss += loss.loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                [self.train_metrics.update(f'train_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            val_loss = self.evaluate(loss_function, network, val_loader,
                                     log_prefix=log_prefix, epoch_id=epoch)
            if val_loss < best_loss:
                filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
                save_model(network, filename)
                best_loss = val_loss

        self.logger.info('Training process has finished.')


class TrainerCrossDomainSparseMoE(TrainerCrossDomainMoe):
    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        loss_function = MoEBCELoss(class_weights=self.config['loss/class_weights'])

        num_workers = self.config['dataset1_train_loader/args/num_workers']

        for i, test_dataset in enumerate(self.datasets):
            train_dataset = [x for j, x in enumerate(self.datasets) if i != j]
            train_dataset = TextDataset(pd.concat(train_dataset, axis=0))

            test_dataset = TextDataset(test_dataset)

            train_data_loader = DataLoader(train_dataset, batch_size=batch_sizes[i],
                                           shuffle=True, num_workers=num_workers)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_sizes[i],
                                           shuffle=False, num_workers=num_workers)

            self.train_on_dataset(train_data_loader, test_data_loader,
                                  loss_function, log_prefix=chr(ord('A') + i))

    def train_on_dataset(self, train_loader, val_loader, loss_function, log_prefix='A'):
        best_loss = np.inf

        num_epochs = self.config['trainer/epochs']
        len_epoch = len(train_loader)

        network = MoESparseModel(self.text_embedding_size,
                                 self.expert_model_dump_by_dataset[log_prefix],
                                 use_hard_prediction=self.config['use_hard_prediction'])
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss_container = loss_function(moe_outputs, targets, network)
                loss = loss_container.loss + moe_outputs.loss
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                [self.train_metrics.update(f'train_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss_container._asdict().items() if metric_name != 'loss']
                self.train_metrics.update(f'train_{log_prefix}/loss', loss.item())

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            val_loss = self.evaluate(loss_function, network, val_loader,
                                     log_prefix=log_prefix, epoch_id=epoch)
            if val_loss < best_loss:
                filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
                save_model(network, filename)
                best_loss = val_loss

        self.logger.info('Training process has finished.')


class TrainerCrossDomainMoeWithDomainSignal(TrainerCrossDomainMoe):
    def _process_datasets(self, dataset_list):
        return [(a.dataset.data, b.dataset.data) for a, b in dataset_list]

    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        batch_sizes = [32] * len(batch_sizes)

        loss_function = MoELossWithDomain(
            lambda_l1_gate=self.config['loss/lambda_l1_gate'],
            lambda_l1_exp=0,
            lambda_balancing=self.config['loss/lambda_balancing'],
            class_weights=self.config['loss/class_weights'])

        for i, test_dataset in enumerate(self.datasets):
            train_dataset = [x[1] for j, x in enumerate(self.datasets) if i != j]
            for j, df in enumerate(train_dataset):
                df['src'] = j
            train_dataset = TextDatasetWithSource(pd.concat(train_dataset, axis=0))

            test_dataset = TextDataset(pd.concat(test_dataset, axis=0))

            train_data_loader = create_fast_dataloader_from_dataset(train_dataset,
                                                                    batch_size=batch_sizes[i],
                                                                    shuffle=True)

            test_data_loader = create_fast_dataloader_from_dataset(test_dataset,
                                                                   batch_size=batch_sizes[i],
                                                                   shuffle=False)

            self.train_on_dataset(train_data_loader, test_data_loader,
                                  loss_function, log_prefix=chr(ord('A') + i))

        print(f'Sim end. Result folder is {self.config.save_dir}')
        output_result_file = self.config.save_dir / "results_macro_f1.xlsx"
        if output_result_file.exists():
            df = pd.read_excel(output_result_file)
            print(df.head(100))

    def train_on_dataset(self, train_loader, val_loader, loss_function, log_prefix='A'):
        best_loss = np.inf

        num_epochs = self.config['trainer/epochs']
        len_epoch = len(train_loader)

        network = MoEWithDomainDetector(
                      self.text_embedding_size,
                      self.expert_model_dump_by_dataset[log_prefix],
                      domain_size=len(self.datasets) - 1,
                      trainable_expert=self.config['trainable_expert'],
                      use_hard_prediction=self.config['use_hard_prediction'])
        network.to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        self.logger.info(f'Model:\n {network}')

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, targets, srcs = data
                inputs, targets, srcs = (inputs.to(self.device),
                                         targets.to(self.device),
                                         srcs.long().to(self.device))

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, srcs, network)
                loss.loss.backward()
                optimizer.step()

                current_loss += loss.loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                [self.train_metrics.update(f'train_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            val_loss = self.evaluate(loss_function, network, val_loader,
                                     log_prefix=log_prefix, epoch_id=epoch)
            if val_loss < best_loss:
                filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
                save_model(network, filename)
                best_loss = val_loss

        self.logger.info('Training process has finished.')

    def evaluate(self, loss_function, network, loader, log_prefix='A', epoch_id=0):
        current_loss = 0.
        network.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, None, network, validation=True)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                [self.valid_metrics.update(f'valid_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                current_loss += loss.loss.item()

                output = moe_outputs.weighted_outputs

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(targets.cpu().numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'valid_{log_prefix}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'valid_{log_prefix}'):
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        # save results to file
        desc_dict = {
            'DATASET': log_prefix,
            'run_name': self.config['name'],
            'run_id': self.config.save_dir.name,
            'epoch': epoch_id
        }
        output_file = self.config.save_dir / 'results.xlsx'
        save_results_to_file(output_file, log, desc_dict)

        if 'metric_validation' in self.config:
            metric_name = self.config["metric_validation"]
            return log[metric_name] if metric_name in log else None
        return current_loss / len(loader)


class TrainerCrossDomainMoeWithModelSelection(TrainerCrossDomainMoe):
    """
    Implement a model selection strategy to use only the BEST models in the MoE.
    It is a pre-train stage where all models are filtered by a criteria evaluated on
    the training set
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.logger.info('Init model selection')
        for i, test_dataset in enumerate(self.datasets):
            dataset_label = chr(ord('A') + i)

            self.logger.info(f'Process dataset: {dataset_label}')
            results = {}

            for j, partition in enumerate(self.datasets):
                if j == i: continue
                partition_label = chr(ord('A') + j)

                train_dataset = TextDataset(partition[1])
                train_data_loader = create_fast_dataloader_from_dataset(train_dataset,
                                                                        batch_size=100,
                                                                        shuffle=False)

                all_results = []
                for model in self.expert_model_dump_by_dataset[dataset_label]:
                    model_label = re.findall(r'model-([A-Z_]+).pth', model.name)[0]
                    if '_' not in model_label:
                        continue

                    metric_result = self._do_model_selection(model, train_data_loader)
                    all_results.append((metric_result, model))

                    if partition_label in results:
                        results[partition_label].append((metric_result, model))
                    else:
                        results[partition_label] = [(metric_result, model)]

                all_results.sort(reverse=True)

                self.logger.info(f'BEST models on {partition_label} for TEST {dataset_label}:')
                for r, f in all_results:
                    self.logger.info(f'{r:.4f}\t{f.name}')

            self.logger.info(f'SELECTED models for TEST {dataset_label}:')
            for model_list in results.values():
                model_list.sort(reverse=True)
                while len(model_list) > 2:
                    del model_list[-1]
                for r, f in model_list:
                    self.logger.info(f'{r:.4f}\t{f.name}')

            self.expert_model_dump_by_dataset[dataset_label] = [x[1]
                                                                for model_list in results.values()
                                                                for x in model_list]

        self.logger.info('init model selection DONE')

    def _process_datasets(self, dataset_list):
        return [(a.dataset.data, b.dataset.data) for a, b in dataset_list]

    def _do_model_selection(self, model_fname, test_data_loader):

        network = BaseClassifier(self.text_embedding_size)
        network.load_state_dict(torch.load(model_fname))
        network.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for inputs, targets in tqdm(test_data_loader):
                y = network(inputs)
                y = torch.sigmoid(y)

                all_outputs.extend((y.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(targets.cpu().numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            report = classification_report_sklearn(all_targets,
                                                   all_outputs,
                                                   target_names=['real', 'fake'],
                                                   output_dict=True)

            return report['macro avg']['f1-score']

    def train(self):
        self.training_step = 0
        batch_sizes = [self.config['dataset1_train_loader/args/batch_size'] or 16,
                       self.config['dataset2_train_loader/args/batch_size'] or 16,
                       self.config['dataset3_train_loader/args/batch_size'] or 16,
                       self.config['dataset4_train_loader/args/batch_size'] or 16,
                       self.config['dataset5_train_loader/args/batch_size'] or 16]

        batch_sizes = [32] * len(batch_sizes)

        loss_function = MoELossWithDomain(
            lambda_l1_gate=self.config['loss/lambda_l1_gate'],
            lambda_l1_exp=0,
            lambda_balancing=self.config['loss/lambda_balancing'],
            class_weights=self.config['loss/class_weights'])

        for i, test_dataset in enumerate(self.datasets):
            train_dataset = [x[1] for j, x in enumerate(self.datasets) if i != j]
            for j, df in enumerate(train_dataset):
                df['src'] = j
            train_dataset = TextDatasetWithSource(pd.concat(train_dataset, axis=0))
            test_dataset = TextDataset(pd.concat(test_dataset, axis=0))

            # train_data_loader = DataLoader(train_dataset, batch_size=batch_sizes[i],
            #                                shuffle=True, num_workers=num_workers)
            # test_data_loader = DataLoader(test_dataset, batch_size=batch_sizes[i],
            #                                shuffle=False, num_workers=num_workers)

            train_data_loader = create_fast_dataloader_from_dataset(train_dataset,
                                                                    batch_size=batch_sizes[i],
                                                                    shuffle=True)

            test_data_loader = create_fast_dataloader_from_dataset(test_dataset,
                                                                   batch_size=batch_sizes[i],
                                                                   shuffle=False)

            self.train_on_dataset(train_data_loader, test_data_loader,
                                  loss_function, log_prefix=chr(ord('A') + i))

        print(f'Sim end. Result folder is {self.config.save_dir}')
        output_result_file = self.config.save_dir / "results_macro_f1.xlsx"
        if output_result_file.exists():
            df = pd.read_excel(output_result_file)
            print(df.head(100))

    def train_on_dataset(self, train_loader, val_loader, loss_function, log_prefix='A'):
        best_loss = np.inf

        num_epochs = self.config['trainer/epochs']
        len_epoch = len(train_loader)

        network = MoEWithDomainDetector(
                      self.text_embedding_size,
                      self.expert_model_dump_by_dataset[log_prefix],
                      domain_size=len(self.datasets) - 1,
                      trainable_expert=self.config['trainable_expert'],
                      use_hard_prediction=self.config['use_hard_prediction'])
        network.to(self.device)

        self._pre_train_domain_detector(network, train_loader, log_prefix)

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        self.logger.info(f'Model:\n {network}')

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, targets, srcs = data
                inputs, targets, srcs = (inputs.to(self.device),
                                         targets.to(self.device),
                                         srcs.long().to(self.device))

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, srcs, network)
                loss.loss.backward()
                optimizer.step()

                current_loss += loss.loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                [self.train_metrics.update(f'train_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/loss_epoch',
                                      current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'train_{log_prefix}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            val_loss = self.evaluate(loss_function, network, val_loader,
                                     log_prefix=log_prefix, epoch_id=epoch)
            if val_loss < best_loss:
                filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
                save_model(network, filename)
                best_loss = val_loss

        self.logger.info('Training process has finished.')

    def evaluate(self, loss_function, network, loader, log_prefix='A', epoch_id=0):
        current_loss = 0.
        network.eval()

        store_raw_data = epoch_id == 0 or epoch_id == 49

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            raw_data = []

            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs, targets, None, network, validation=True)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                [self.valid_metrics.update(f'valid_{log_prefix}/{metric_name}', value.item())
                 for metric_name, value in loss._asdict().items()]

                current_loss += loss.loss.item()

                output = moe_outputs.weighted_outputs

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(targets.cpu().numpy().astype(int))

                if store_raw_data:
                    y = moe_outputs.weighted_outputs.detach().cpu().numpy()
                    experts_outputs = moe_outputs.experts_outputs.detach().cpu().numpy()
                    gate = moe_outputs.gate_probs.detach().cpu().numpy()
                    domain = moe_outputs.domain_outputs.detach().cpu().numpy()
                    targets = targets.cpu().numpy().astype(int)
                    for i in range(output.shape[0]):
                        row = y[i].tolist() + experts_outputs[i].flatten().tolist() \
                            + gate[i].tolist() + domain[i].tolist() + [targets[i].tolist()]
                        raw_data.append(row)

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'valid_{log_prefix}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'valid_{log_prefix}'):
                self.logger.info('    {:15s}: {}'.format(str(key), value))

        # save results to file
        desc_dict = {
            'DATASET': log_prefix,
            'run_name': self.config['name'],
            'run_id': self.config.save_dir.name,
            'epoch': epoch_id
        }
        output_file = self.config.save_dir / 'results.xlsx'
        save_results_to_file(output_file, log, desc_dict)

        if store_raw_data:
            num_expert = network.num_experts
            columns = ['Y']
            for i in range(num_expert):
                columns.append(f'E{i+1}')
            for i in range(num_expert):
                columns.append(f'G{i+1}')
            for i in range(4):  # domain are fixed
                columns.append(f'D{i+1}')
            columns.append('T')
            output_file = self.config.save_dir / f'raw_data_{log_prefix}_{epoch_id:03d}.xlsx'
            pd.DataFrame(raw_data, columns=columns).to_excel(output_file)

        if 'metric_validation' in self.config:
            metric_name = self.config["metric_validation"]
            return log[metric_name] if metric_name in log else None
        return current_loss / len(loader)

    def _pre_train_domain_detector(self, network, train_loader, log_prefix):
        num_epochs = 20
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        loss_function = torch.nn.CrossEntropyLoss()
        len_epoch = len(train_loader)

        self.logger.info(f'Pre-Train domain detector for run {log_prefix}')

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            network.train()

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(train_loader, 0):
                inputs, _, srcs = data
                inputs, srcs = inputs.to(self.device), srcs.long().to(self.device)

                optimizer.zero_grad()
                moe_outputs = network(inputs)
                loss = loss_function(moe_outputs.domain_outputs, srcs)
                loss += .1 * torch.sum(.5 - torch.abs(.5 - moe_outputs.domain_outputs))
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                batch_number += 1
                self.training_step += 1
                self.train_metrics.update(f'train_{log_prefix}/domain_pre_train_loss',
                                          loss.item())

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Domain - Train Epoch: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'train_{log_prefix}/domain_pre_train_loss_epoch',
                                      current_loss / batch_number)

        self.logger.info('Domain - Training process has finished.')

