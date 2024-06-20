import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils import resample
from torch.nn import CrossEntropyLoss
from torcheval.metrics.functional import r2_score as r2_score_function

import data_loader.data_loaders as module_data
from data_loader.data_loaders import DomainDataset, MultiDataset, TextDataDataset
from logger import TensorboardWriter
from model.loss import TextClassifierModelTrainLoss, FocalLoss, BinaryCrossEntropy, AutoEncoderMSE, \
    CrossDomainModelAutoEncoderBasedLoss, AutoEncoderGeneric, \
    CrossDomainModelAutoEncoderBasedLossR2, AutoEncoderAdv, CrossDomainModelAutoEncoderWithFLLoss
from model.metric import get_label_classification_report
from model.model import DomainClassifier, TextClassifierWithFeatureExtractionModel, \
    TextClassifierModel, \
    CrossDomainAutoEncoderModel, NoisedVAE, DomainAutoencoderAndMapper, DomainClassifierLarge, \
    TextClassifierDebugModel, \
    DomainMultiTargetAndMapper, MultiVAE, DomainAutoencoder
from utils import MetricTracker

from time import time

import pickle
import hashlib


def reset_weights(m):
  """
    Try resetting model weights to avoid
    weight leakage.
  """
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()


def save_model(model, f_name):
    print('save model to', f_name)
    current_device = next(model.parameters()).device
    torch.save(model.cpu().state_dict(), f_name)
    model.to(current_device)


class TrainerDualArch:
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.training_step = 0
        self.train_metrics = None
        self.valid_metrics = None
        self.test_metrics = None

        self.init_metrics()

    def init_metrics(self):
        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/discriminator_erica' for c in 'AB' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        m1 += [f'test/discriminator_erica_{i}' for i in 'AB']
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          *m1, writer=self.writer)

    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        # self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator.eval()

        loss_function = TextClassifierModelTrainLoss()

        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        # TEST
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

    def build_domain_discriminator(self, batch_size, num_epochs=40):
        dataset = DomainDataset(self.dataset1, self.dataset2)

        # model = DomainClassifierLarge(self.text_embedding_size).to(self.device)
        model = DomainClassifier(self.text_embedding_size).to(self.device)
        model.train()
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # loss_function = torch.nn.BCELoss()
        loss_function = FocalLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            count_correct = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1

                count_correct += ((outputs >= .5).flatten() == targets.flatten()).sum()

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            accuracy_on_epoch = count_correct / len(dataset)
            self.logger.info(f'Accuracy on epoch {epoch} is {accuracy_on_epoch:.4f}')
            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            log['accuracy_on_epoch'] = accuracy_on_epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain Discriminator process has finished.')

        model.load_state_dict(best_network_weights)
        model.eval()
        return model

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=10, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            network = TextClassifierWithFeatureExtractionModel(self.text_embedding_size).to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0]
            batch_number = 0
            all_loss_dfx = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                y, f_x = network(inputs)

                d_f_x = self.model_discriminator(f_x)

                loss, classifier_loss, loss_dfx = loss_function(y, targets, d_f_x)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()
                current_loss[1] += classifier_loss.item()
                current_loss[2] += loss_dfx.item()
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/discriminator_erica', np.mean(all_loss_dfx))

            loss_function.update_alphas(current_loss[1] / batch_number,
                                        current_loss[2] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            all_loss_dfx = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output, f_x = network(data)
                d_f_x = self.model_discriminator(f_x)
                loss, classifier_loss, loss_dfx = loss_function(output, target.to(self.device), d_f_x)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())

                current_loss += loss.item()

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)
            self.valid_metrics.update(f'fold_{log_prefix}_{fold}/discriminator_erica', np.mean(all_loss_dfx))

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            all_loss_dfx = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output, f_x = network(data)
                d_f_x = self.model_discriminator(f_x)
                loss, classifier_loss, loss_dfx = loss_function(output, target.to(self.device), d_f_x)

                all_loss += loss.item()
                batch_num += 1

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)
        self.test_metrics.update(f'test/discriminator_erica_{log_prefix}', np.mean(all_loss_dfx))

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)



class TrainerDualArchWithAutoEncoder:
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, data_loader3=None, data_loader4=None, lr_scheduler=None, len_epoch=None, **kargs):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.dataset3 = data_loader3.dataset
        if data_loader4 is not None:
            self.dataset4 = data_loader4.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'ABCD' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_term1' for c in 'ABCD' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_term2' for c in 'ABCD' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'ABCD' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'ABCD' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/r2score' for c in 'ABCD' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}{j}_{m}' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for m in get_label_classification_report() if i != j]
        m1 += [f'test/{i}{j}_r2score' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] if i != j]
        self.test_metrics = MetricTracker('test/loss_AB', 'test/loss_AC', 'test/loss_AD',
                                          'test/loss_BA', 'test/loss_BC', 'test/loss_BD',
                                          'test/loss_CA', 'test/loss_CB', 'test/loss_CD',
                                          'test/loss_DA', 'test/loss_DB', 'test/loss_DC',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator.eval()

        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        # loss_function = CrossDomainModelAutoEncoderWithFLLoss(gamma=2.)
        start = time()
        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')
        # d_model1 = pickle.dumps(model1)
        # hashId = hashlib.md5()
        # hashId.update(repr(d_model1).encode('utf-8'))
        # print(hashId.hexdigest())
        print(f'Elapsed time: {time() - start}')

        loss_function = CrossDomainModelAutoEncoderBasedLoss(class_weights=[.6, .4])
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        start = time()
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')
        print(f'Elapsed time: {time() - start}')

        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        batch_size_3 = self.config['data_loader_valid/args/batch_size'] or 16
        start = time()
        model3 = self.train_on_dataset(self.dataset3, batch_size_3, loss_function, log_prefix='C')
        print(f'Elapsed time: {time() - start}')

        # loss_function = CrossDomainModelAutoEncoderBasedLoss()
        # batch_size_4 = self.config['data_loader_valid/args/batch_size'] or 16
        # start = time()
        # model4 = self.train_on_dataset(self.dataset4, batch_size_4, loss_function, log_prefix='D')
        # print(f'Elapsed time: {time() - start}')

        # TEST
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='AB')
        self.test(self.dataset3, model1, batch_size_3, loss_function, log_prefix='AC')
        # self.test(self.dataset4, model1, batch_size_4, loss_function, log_prefix='AD')

        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='BA')
        self.test(self.dataset3, model2, batch_size_3, loss_function, log_prefix='BC')
        # self.test(self.dataset4, model2, batch_size_4, loss_function, log_prefix='BD')

        self.test(self.dataset1, model3, batch_size_1, loss_function, log_prefix='CA')
        self.test(self.dataset2, model3, batch_size_2, loss_function, log_prefix='CB')
        # self.test(self.dataset4, model3, batch_size_4, loss_function, log_prefix='CD')

        # self.test(self.dataset1, model4, batch_size_1, loss_function, log_prefix='DA')
        # self.test(self.dataset2, model4, batch_size_2, loss_function, log_prefix='DB')
        # self.test(self.dataset3, model4, batch_size_3, loss_function, log_prefix='DC')

    def build_domain_discriminator(self, batch_size, num_epochs=20):
        # Undersampling
        # train1, _ = train_test_split(self.dataset1, train_size=0.1, stratify=self.dataset1.data.label)
        train1, _ = train_test_split(self.dataset1, train_size=0.2, stratify=self.dataset1.data.label)
        # train3, _ = train_test_split(self.dataset3, train_size=0.022, stratify=self.dataset3.data.label)
        train3, _ = train_test_split(self.dataset3, train_size=0.045, stratify=self.dataset3.data.label)

        # train4, _ = train_test_split(self.dataset4, train_size=0.012, stratify=self.dataset4.data.label)

        dataset = DomainDataset(train1, self.dataset2, train3)  # self.dataset3, self.dataset4, train4

        # Oversampling
        # n_samples = 10000
        # train1 = resample(self.dataset1, n_samples=n_samples, replace=True, stratify=self.dataset1.data.label,
        #                   random_state=0)
        # train2 = resample(self.dataset2, n_samples=n_samples, replace=True, stratify=self.dataset2.data.label,
        #                   random_state=0)
        # train3, _ = train_test_split(self.dataset3, train_size=0.481, stratify=self.dataset3.data.label)
        # dataset = DomainDataset(train1, train2, train3)  # self.dataset3

        # tmp = 'dataset/dataset_fakenewskaggle_textemb_deberta.parquet'
        # dataset = module_data.TextDataset(tmp)

        # tmp = 'dataset/dataset_ncd_textemb_deberta.parquet'
        # dataset = module_data.TextDataset(tmp)

        # model = DomainAutoencodeAndMapper(
        #         input_size=self.text_embedding_size,
        #         p_dims=[64, 256, self.text_embedding_size])\
        #     .to(self.device)

        # tmp = 'dataset/amz_reviews_processed_deberta_emb.parquet'
        # dataset = module_data.ModelRatingsDataset(tmp)

        # p_dims=[128, 512, self.text_embedding_size

        model = DomainAutoencoderAndMapper(
            input_size=self.text_embedding_size,
            p_dims=[384, 512, self.text_embedding_size]) \
            .to(self.device)
        self.logger.info('Discriminator Model' + str(model))
        model.train()
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        loss_function = AutoEncoderMSE(is_vae_model=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, _ = data
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                y, mu, logvar = model(inputs, use_output_activation=False)

                loss = loss_function(y, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain AE process has finished.')

        model.load_state_dict(best_network_weights)
        model.eval()

        filename = str(self.checkpoint_dir / f'discriminator.pth')
        save_model(model, filename)
        return model

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=20, log_prefix='A'):
        p_rate = 0.5

        # if len(dataset.data) < 1000:
        #     p_rate = 0.3

        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            emb_size = self.text_embedding_size
            network = CrossDomainAutoEncoderModel(f_x_layers=(emb_size, 256, 64, emb_size),
                                                  g_layers=(emb_size, 256, 1), use_dropout=True, p_rate=p_rate)\
                .to(self.device)
            # network.f_x = copy.deepcopy(self.model_discriminator.ae.q_layers)
            # network.f_x = nn.Sequential(*network.f_x)

            if fold == 0 and log_prefix == 'A':
                self.logger.info('Classifier Model' + str(network))
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader,
                                log_prefix=log_prefix)
            valid_metric = self._test_on_fold(fold, loss_function, network, testloader,
                                log_prefix=log_prefix)

            if valid_metric is not None:
                if 'metric_validation_target' in self.config:
                    criteria = self.config['metric_validation_target'].lower()

                    if criteria == 'max':
                        improved = valid_metric > best_loss
                    else:
                        improved = valid_metric < best_loss
                else:
                    improved = valid_metric < best_loss

                if improved:
                    best_loss = valid_metric
                    best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0]
            batch_number = 0

            r2_score_aggregator = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                x = self.model_discriminator.get_mapping(inputs)
                y, f_x = network(x)

                d_f_x, _, _ = self.model_discriminator(f_x, False)
                z, _, _ = self.model_discriminator(inputs)

                loss, loss_bce, loss_reg_term = loss_function(y, targets, d_f_x, z)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()
                current_loss[1] += loss_bce.item()
                current_loss[2] += loss_reg_term.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                self.train_metrics.update(
                    f'fold_{log_prefix}_{fold}/loss_term1', loss_bce.item())
                self.train_metrics.update(
                    f'fold_{log_prefix}_{fold}/loss_term2', loss_reg_term.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch',
                                      current_loss[0] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/r2score',
                                      np.mean(r2_score_aggregator))

            loss_function.update_alphas(current_loss[1] / batch_number,
                                        current_loss[2] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                x = self.model_discriminator.get_mapping(data)
                output, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x, False)
                z, _, _ = self.model_discriminator(data)

                loss, loss_bce, loss_reg_term = loss_function(output, target.to(self.device), d_f_x, z)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                self.valid_metrics.update(
                    f'fold_{log_prefix}_{fold}/loss_term1', loss_bce.item())
                self.valid_metrics.update(
                    f'fold_{log_prefix}_{fold}/loss_term2', loss_reg_term.item())

                current_loss += loss.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)
            self.valid_metrics.update(f'fold_{log_prefix}_{fold}/r2score', np.mean(r2_score_aggregator))

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        # 'fold_A_3/metrics/macro_avg_f1-score'
        if 'metric_validation' in self.config:
            metric_name = self.config["metric_validation"]
            metric_name = f'fold_{log_prefix}_{fold}/metrics/{metric_name}'
            return log[metric_name] if metric_name in log else None
        return current_loss / len(testloader)

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                x = self.model_discriminator.get_mapping(data)
                output, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x, False)
                z, _, _ = self.model_discriminator(data)

                loss, loss_bce, loss_reg_term = loss_function(output, target.to(self.device), d_f_x, z)

                all_loss += loss.item()
                batch_num += 1

                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)
        self.test_metrics.update(f'test/{log_prefix}_r2score', np.mean(r2_score_aggregator))

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerMultiDualArchWithAutoEncoder(TrainerDualArchWithAutoEncoder):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, data_loader3=None, data_loader4=None, lr_scheduler=None, len_epoch=None, **kargs):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.dataset3 = data_loader3.dataset
        if data_loader4 is not None:
            self.dataset4 = data_loader4.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}{j}_{i}/loss' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += [f'fold_{c}{j}_{i}/loss_term1' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += [f'fold_{c}{j}_{i}/loss_term2' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_{c}{j}_{i}/loss_epoch' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += [f'fold_{c}{j}_{i}/metrics/{m}' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}{j}_{i}/r2score' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}{j}{k}_{m}' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for k in ['A', 'B', 'C', 'D'] for m in get_label_classification_report() if i != j]
        m1 += [f'test/{i}{j}{k}_r2score' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for k in ['A', 'B', 'C', 'D'] if i != j]
        self.test_metrics = MetricTracker('test/loss_ABC', 'test/loss_ACB', 'test/loss_BCA',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        sample_perc = self.config['sample_perc']
        self.logger.info('Run with sample perc=' + str(sample_perc))
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        batch_size_3 = self.config['data_loader_train3/args/batch_size'] or 16
        self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator.eval()

        # TRAIN ON GOSSIP + POLITIFACT + FNKaggle, TEST on FNKaggle
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        train3, test3 = train_test_split(self.dataset3.data, train_size=0.1, stratify=self.dataset3.data.label,
                                         random_state=123)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset3.data))
            train3, _ = train_test_split(train3, train_size=n, stratify=train3.label, random_state=123)

        start = time()
        model1 = self.train_on_dataset(MultiDataset(self.dataset1, self.dataset2, train3), batch_size_1,
                                       loss_function, log_prefix='AB', num_epochs=30)
        print(f'Elapsed time: {time() - start}')

        self.test(TextDataDataset(test3), model1, batch_size_3, loss_function, log_prefix='ABC')

        # TRAIN ON GOSSIP + FNKaggle + POLITIFACT, TEST on POLITIFACT
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        train2, test2 = train_test_split(self.dataset2.data, train_size=0.1, stratify=self.dataset2.data.label,
                                         random_state=123)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset2.data))
            train2, _ = train_test_split(train2, train_size=n, stratify=train2.label, random_state=123)

        start = time()
        model2 = self.train_on_dataset(MultiDataset(self.dataset1, train2, self.dataset3), batch_size_3,
                                       loss_function, log_prefix='AC', num_epochs=30)
        print(f'Elapsed time: {time() - start}')

        self.test(TextDataDataset(test2), model2, batch_size_2, loss_function, log_prefix='ACB')

        # TRAIN ON POLITIFACT + FNKaggle + GOSSIP, TEST on GOSSIP
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        train1, test1 = train_test_split(self.dataset1.data, train_size=0.1, stratify=self.dataset1.data.label)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset1.data))
            train1, _ = train_test_split(train1, train_size=n, stratify=train1.label, random_state=123)

        start = time()
        model3 = self.train_on_dataset(MultiDataset(train1, self.dataset2, self.dataset3), batch_size_3,
                                       loss_function, log_prefix='BC', num_epochs=30)
        print(f'Elapsed time: {time() - start}')
        self.test(TextDataDataset(test1), model3, batch_size_1, loss_function, log_prefix='BCA')


class TrainerDualArchWithAutoEncoderLossAdv(TrainerDualArchWithAutoEncoder):
    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator.eval()

        loss_function = CrossDomainModelAutoEncoderBasedLoss()

        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        loss_function = CrossDomainModelAutoEncoderBasedLoss(class_weights=[.7, .3])
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        # TEST
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

class TrainerDualArchOnMixedDataset:
    """
    Trainer class for model trained and tested on one dataset
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.data_loader, self.valid_data_loader = data_loader, valid_data_loader
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_all_{i}/loss' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_all_{i}/loss_epoch' for i in range(self.k_fold)]
        m += [f'fold_all_{i}/discriminator_erica' for i in range(self.k_fold)]
        m += [f'fold_all_{i}/metrics/{m}' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/all_{m}' for m in get_label_classification_report()]
        m1 += [f'test/discriminator_erica']
        self.test_metrics = MetricTracker('test/loss_all', *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        self.model_discriminator = self.build_domain_discriminator(batch_size_1)
        self.model_discriminator.eval()

        loss_function = TextClassifierModelTrainLoss()

        model = self.train_on_dataset(self.data_loader.dataset, batch_size_1, loss_function, log_prefix='all')

        self.test(self.valid_data_loader.dataset, model, batch_size_1, loss_function, log_prefix='all')

    def build_domain_discriminator(self, batch_size, num_epochs=50):
        model = DomainClassifier().to(self.device)
        model.train()
        trainloader = self.data_loader

        loss_function = BinaryCrossEntropy()
        # loss_function = FocalLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            count_correct = 0
            count_all = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, _, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1

                count_correct += ((outputs >= .5).flatten() == targets.flatten()).sum()
                count_all += outputs.shape[0]

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            accuracy_on_epoch = count_correct / count_all
            self.logger.info(f'Accuracy on epoch {epoch} is {accuracy_on_epoch:.4f}')
            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            log['accuracy_on_epoch'] = accuracy_on_epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain Discriminator process has finished.')

        model.load_state_dict(best_network_weights)
        model.eval()
        return model

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=10, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            network = TextClassifierWithFeatureExtractionModel().to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0]
            batch_number = 0
            all_loss_dfx = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets, _ = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                y, f_x = network(inputs)

                d_f_x = self.model_discriminator(f_x)

                loss, classifier_loss, loss_dfx = loss_function(y, targets, d_f_x)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()
                current_loss[1] += classifier_loss.item()
                current_loss[2] += loss_dfx.item()
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/discriminator_erica', np.mean(all_loss_dfx))

            loss_function.update_alphas(current_loss[1] / batch_number,
                                        current_loss[2] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            all_loss_dfx = []

            for batch_idx, (data, target, _) in enumerate(testloader):
                data = data.to(self.device)

                output, f_x = network(data)
                d_f_x = self.model_discriminator(f_x)
                loss, classifier_loss, loss_dfx = loss_function(output, target.to(self.device), d_f_x)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())

                current_loss += loss.item()
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)
            self.valid_metrics.update(f'fold_{log_prefix}_{fold}/discriminator_erica', np.mean(all_loss_dfx))

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            all_loss_dfx = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output, f_x = network(data)
                d_f_x = self.model_discriminator(f_x)
                loss, classifier_loss, loss_dfx = loss_function(output, target.to(self.device), d_f_x)

                all_loss += loss.item()
                batch_num += 1

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))
                all_loss_dfx.extend(torch.abs(.5 - d_f_x).cpu().detach().numpy())

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)
        self.test_metrics.update(f'test/discriminator_erica', np.mean(all_loss_dfx))

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerSimpleClassifierOnMixedDataset:
    """
    Trainer class for model trained and tested on one dataset
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.data_loader, self.valid_data_loader = data_loader, valid_data_loader
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_all_{i}/loss' for i in range(self.k_fold)]
        m += [f'fold_all_{i}/loss_epoch' for i in range(self.k_fold)]
        m += [f'fold_all_{i}/metrics/{m}' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/all_{m}' for m in get_label_classification_report()]
        self.test_metrics = MetricTracker('test/loss_all', *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16

        loss_function = BinaryCrossEntropy()

        model = self.train_on_dataset(self.data_loader.dataset, batch_size_1, loss_function, log_prefix='all')

        self.test(self.valid_data_loader.dataset, model, batch_size_1, loss_function, log_prefix='all')

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=10, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            network = TextClassifierModel().to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                y = network(inputs)

                loss = loss_function(y, targets)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, target.to(self.device))

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())

                current_loss += loss.item()

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, target.to(self.device))

                all_loss += loss.item()
                batch_num += 1

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)



class TrainerDualArchWithGPretrained(TrainerDualArch):

    def init_metrics(self):
        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/loss_g_x_pretrain' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/discriminator_erica' for c in 'AB' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        m1 += [f'test/discriminator_erica_{i}' for i in 'AB']
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          *m1, writer=self.writer)

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=10, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # pretrain G(x)
            input_dim = 768
            fx = torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim // 4),
                torch.nn.Tanh(),
                torch.nn.Linear(input_dim // 4, input_dim),
            ).to(self.device)

            self._pretrain_fx(fold, fx, trainloader, log_prefix=log_prefix)

            # Init the neural network
            network = TextClassifierWithFeatureExtractionModel(input_dim)
            network._fx = fx
            network = network.to(self.device)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model


    def _pretrain_fx(self, fold, network, trainloader, num_epochs=10, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        loss_function = torch.nn.MSELoss()

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                y = network(inputs)

                loss = loss_function(y, inputs)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_g_x_pretrain', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('PreTrain G(x) Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

        self.logger.info('Pre Training G(x) process has finished.')


class TrainerDebug:
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.training_step = 0
        self.train_metrics = None
        self.valid_metrics = None
        self.test_metrics = None

        self.init_metrics()

    def init_metrics(self):
        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch']
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/discriminator_erica' for c in 'AB' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        m1 += [f'test/discriminator_erica_{i}' for i in 'AB']
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          *m1, writer=self.writer)

    def train(self):
        self.training_step = 0

        loss_function = BinaryCrossEntropy()

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        # TEST
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=10, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            network = TextClassifierDebugModel(self.text_embedding_size).to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0]
            batch_number = 0
            all_loss_dfx = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                y = network(inputs)

                loss = loss_function(y, targets)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, target.to(self.device))

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())

                current_loss += loss.item()

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, target.to(self.device))

                all_loss += loss.item()
                batch_num += 1

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerDualArchWithAutoEncoderPretrained:
    """
    The domain model is pretrained on a specialized task then is fine tuned on the reconstruction task
    while training the whole network
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # domain data loader
        domain_dataset = config.init_obj('dataset_pretrain', module_data)
        self.domain_data_loader = config.init_obj('data_loader_pretrain', module_data, dataset=domain_dataset)

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_term1' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_term2' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_domain' for c in 'AB' for i in range(self.k_fold)]
        m += ['domain_model/loss_domain', 'domain_model/loss_domain_epoch', 'domain_model/accuracy_epoch']
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/r2score' for c in 'AB' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        m1 += [f'test/{i}_r2score' for i in 'AB']
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16

        self.model_discriminator = self.build_domain_discriminator()

        # loss_function = CrossDomainModelAutoEncoderBasedLoss(class_weights=[.7, .3])
        loss_function = CrossDomainModelAutoEncoderBasedLoss()

        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        loss_function = CrossDomainModelAutoEncoderBasedLoss(class_weights=[.7, .3])

        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        # TEST
        loss_function = CrossDomainModelAutoEncoderBasedLoss()
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

    def build_domain_discriminator(self, num_epochs=10):
        model = DomainMultiTargetAndMapper(
            input_size=self.text_embedding_size,
            p_dims=[64, 32, 6],  # ratings from 0 to 5
            q_dims=[self.text_embedding_size, 256, 64]) \
            .to(self.device)
        self.logger.info('Discriminator Model' + str(model))
        model.train()
        trainloader = self.domain_data_loader

        loss_function = AutoEncoderGeneric(is_vae_model=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0
            all_labels, ok_labels = 0, 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, target = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                y, mu, logvar = model(inputs)

                loss = loss_function(y, target.flatten(), mu, logvar)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1
                all_labels += target.shape[0]
                ok_labels += (torch.argmax(y, dim=1) == target.flatten()).sum().item()

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            self.train_metrics.update('domain_model/accuracy_epoch', ok_labels / all_labels)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain AE process has finished.')

        model.load_state_dict(best_network_weights)
        return model

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=20, log_prefix='A'):
        k_fold = StratifiedKFold(n_splits=self.k_fold, shuffle=True)
        y = [i.item() for _, i in dataset]

        best_loss = np.inf
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset, y)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            emb_size = self.text_embedding_size
            network = CrossDomainAutoEncoderModel(f_x_layers=(emb_size, 156, 256, emb_size),
                                                  g_layers=(emb_size, 256, 256, 1))\
                .to(self.device)
            if fold == 0 and log_prefix == 'A':
                self.logger.info('Classifier Model' + str(network))
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        assert best_model is not None
        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        self.model_discriminator.eval()
        # loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0]
            batch_number = 0

            r2_score_aggregator = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                x = self.model_discriminator.get_mapping(inputs)
                y, f_x = network(x)

                # d_f_x = self.model_discriminator.encode(f_x, use_mapper=False)
                # z = self.model_discriminator.encode(x, use_mapper=False)
                d_f_x, _, _ = self.model_discriminator(f_x, use_mapper=False)
                z, _, _ = self.model_discriminator(x, use_mapper=False)

                loss, loss_bce, loss_reg_term = loss_function(y, targets, d_f_x, z)
                assert not torch.isnan(loss)
                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()
                # current_loss[1] += losses.item()
                # current_loss[2] += loss_dfx.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_term1', loss_bce.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_term2', loss_reg_term.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/r2score', np.mean(r2_score_aggregator))

            # loss_function.update_alphas(current_loss[1] / batch_number,
            #                             current_loss[2] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.model_discriminator.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                x = self.model_discriminator.get_mapping(data)
                output, f_x = network(x)
                d_f_x = self.model_discriminator.encode(f_x, False)
                z = self.model_discriminator.encode(x, False)

                loss, loss_bce, loss_reg_term = loss_function(output, target.to(self.device), d_f_x, z)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss_term1', loss_bce.item())
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss_term2', loss_reg_term.item())

                current_loss += loss.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)
            self.valid_metrics.update(f'fold_{log_prefix}_{fold}/r2score', np.mean(r2_score_aggregator))

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                x = self.model_discriminator.get_mapping(data)
                output, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x, False)
                z, _, _ = self.model_discriminator(data)

                loss, *args = loss_function(output, target.to(self.device), d_f_x, z)

                all_loss += loss.item()
                batch_num += 1

                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)
        self.test_metrics.update(f'test/{log_prefix}_r2score', np.mean(r2_score_aggregator))

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerDualArchWithAutoEncoderPretrainedRecon(TrainerDualArchWithAutoEncoderPretrained):
    def train(self):
        self.training_step = 0

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        self.model_discriminator = self.build_domain_discriminator()

        # loss_function = CrossDomainModelAutoEncoderBasedLoss(class_weights=[.7, .3])
        loss_function = CrossDomainModelAutoEncoderBasedLossR2()
        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        loss_function = CrossDomainModelAutoEncoderBasedLossR2(class_weights=[.7, .3])
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        # TEST
        loss_function = CrossDomainModelAutoEncoderBasedLossR2()
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

    def build_domain_discriminator(self, num_epochs=10):
        model = DomainMultiTargetAndMapper(
            p_dims=[32, 256, self.text_embedding_size],
            input_size=self.text_embedding_size,
            final_activation=None
            ) \
            .to(self.device)
        self.logger.info('Discriminator Model' + str(model))
        model.train()
        trainloader = self.domain_data_loader

        loss_function = AutoEncoderMSE(is_vae_model=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, target = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                y, mu, logvar = model(inputs)

                loss = loss_function(y, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain AE process has finished.')

        model.load_state_dict(best_network_weights)
        model.eval()
        return model


class TrainerDualArchWithAutoEncoderNOCV(TrainerDualArchWithAutoEncoder):
    """
    Trainer without cross validation protocol
    """
    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=20, log_prefix='A'):
        fold = 0
        # best_loss = np.inf
        # best_model = None

        self.logger.info('Running train on full dataset')

        labels = [int(y.item()) for _, y in dataset]
        assert np.unique(labels) == 2, f'fail computing labels array. Unique labels are {np.unique(labels)}'
        train_ids, test_ids = train_test_split(range(len(dataset)),
                                               test_size=.3,
                                               stratify=labels,
                                               random_state=self.config['seed'] or 8165)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        emb_size = self.text_embedding_size
        network = CrossDomainAutoEncoderModel(f_x_layers=(emb_size, 256, emb_size), g_layers=(emb_size, 256, 1))\
            .to(self.device)

        self.logger.info('Model' + str(network))

        self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
        valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     best_model = network

        # return best_model
        return network


class TrainerDualArchWithOnlineAutoEncoder:
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] if 'text_embedding_size' in self.config else 768
        self.logger.info(f'{self.text_embedding_size=}')

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.model_discriminator = None

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_domain' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_domain_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/r2score' for c in 'AB' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        m1 += [f'test/{i}_r2score' for i in 'AB']
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0
        self.model_discriminator = MultiVAE(
            p_dims=[8, 64, 256, self.text_embedding_size]) \
            .to(self.device)
        self.logger.info('Discriminator Model' + str(self.model_discriminator))

        loss_function = CrossDomainModelAutoEncoderBasedLossR2()

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')
        discriminator1 = self.model_discriminator.state_dict()

        loss_function = CrossDomainModelAutoEncoderBasedLossR2(class_weights=[.7, .3])
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')
        discriminator2 = self.model_discriminator.state_dict()

        # TEST
        loss_function = CrossDomainModelAutoEncoderBasedLossR2()
        self.model_discriminator.load_state_dict(discriminator1)
        self.model_discriminator.eval()
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='A')

        self.model_discriminator.load_state_dict(discriminator2)
        self.model_discriminator.eval()
        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='B')

    def build_domain_discriminator(self, batch_size, num_epochs=20):
        dataset = DomainDataset(self.dataset1, self.dataset2)

        # model = DomainAutoencodeAndMapper(
        #         input_size=self.text_embedding_size,
        #         p_dims=[64, 256, self.text_embedding_size])\
        #     .to(self.device)
        model = DomainAutoencoderAndMapper(
            input_size=self.text_embedding_size,
            p_dims=[8, 64, 256, self.text_embedding_size]) \
            .to(self.device)
        self.logger.info('Discriminator Model' + str(model))
        model.train()
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        loss_function = AutoEncoderMSE(is_vae_model=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        best_loss = np.inf
        best_network_weights = None

        for epoch in range(0, num_epochs):
            current_loss = 0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, _ = data
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                y, mu, logvar = model(inputs, use_output_activation=False)

                loss = loss_function(y, inputs, mu, logvar)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'domain_model/loss_domain', loss.item())
                batch_number += 1

                if best_loss > loss.item():
                    best_loss = loss.item()
                    best_network_weights = model.state_dict()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Domain: {} ({}%) Loss: {:.6f}'.format(
                        epoch,
                        f'{(batch_idx + 1) / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update('domain_model/loss_domain_epoch', current_loss / batch_number)
            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if not key.startswith('fold'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training Domain AE process has finished.')

        model.load_state_dict(best_network_weights)
        model.eval()
        return model

    def train_on_dataset(self, dataset, batch_size, loss_function, num_epochs=20, log_prefix='A'):
        k_fold = KFold(n_splits=self.k_fold, shuffle=True)

        best_loss = np.inf
        best_model = None
        best_discriminator_weights = None

        for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

            self.logger.info(f'FOLD {fold}')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            emb_size = self.text_embedding_size
            network = CrossDomainAutoEncoderModel(f_x_layers=(emb_size, 256, emb_size), g_layers=(emb_size, 256, 1))\
                .to(self.device)
            if fold == 0 and log_prefix == 'A':
                self.logger.info('Classifier Model' + str(network))
            network.apply(reset_weights)
            self.model_discriminator.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network
                best_discriminator_weights = self.model_discriminator.state_dict()

        self.model_discriminator.load_state_dict(best_discriminator_weights)
        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        optimizerDiscriminator = torch.optim.Adam(self.model_discriminator.parameters(), lr=1e-3)
        len_epoch = len(trainloader)

        # loss_function.reset()
        loss_discriminator = AutoEncoderAdv()

        for epoch in range(0, num_epochs):
            current_loss = [0, 0, 0, 0]
            batch_number = 0

            r2_score_aggregator = []

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                x, targets = data
                x, targets = x.to(self.device), targets.to(self.device)

                # DISCRIMINATOR
                optimizerDiscriminator.zero_grad()

                z, mu, logvar = self.model_discriminator(x)
                _, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x.detach())
                discr_loss_value = loss_discriminator(z, x, mu, logvar, d_f_x, f_x.detach())

                discr_loss_value.backward()
                optimizerDiscriminator.step()

                # CLASSIFIER
                optimizer.zero_grad()

                y, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x)
                loss, loss_bce, loss_reg_term = loss_function(y, targets, d_f_x.detach(), f_x)

                loss.backward()
                optimizer.step()

                current_loss[0] += loss.item()
                # current_loss[1] += losses.item()
                # current_loss[2] += loss_dfx.item()
                current_loss[3] += discr_loss_value.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_domain', discr_loss_value.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch * 100:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_domain_epoch', current_loss[3] / batch_number)
            self.train_metrics.update(f'fold_{log_prefix}_{fold}/r2score', np.mean(r2_score_aggregator))

            # loss_function.update_alphas(current_loss[1] / batch_number,
            #                             current_loss[2] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')
        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(network.state_dict(), save_path)

    def _test_on_fold(self, fold, loss_function, network, testloader, log_prefix='A'):
        """
        Validate after training an epoch

        :return: the loss value
        """
        current_loss = .0
        network.eval()
        self.model_discriminator.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                x = data.to(self.device)

                output, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x)
                z, _, _ = self.model_discriminator(x)

                loss, loss_bce, loss_reg_term = loss_function(output, target.to(self.device), d_f_x, z)

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.item())

                current_loss += loss.item()
                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

            self.valid_metrics.process_metrics(self.metric_ftns,
                                               f'fold_{log_prefix}_{fold}/metrics/',
                                               all_outputs, all_targets)
            self.valid_metrics.update(f'fold_{log_prefix}_{fold}/r2score', np.mean(r2_score_aggregator))

        # add histogram of model parameters to the tensorboard
        for name, p in network.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        for key, value in log.items():
            if key.startswith(f'fold_{log_prefix}_{fold}'):
                 self.logger.info('    {:15s}: {}'.format(str(key), value))

        return current_loss

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        # m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        # self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B', *m1, writer=self.writer)
        all_loss = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            r2_score_aggregator = []

            for batch_idx, (data, target) in enumerate(testloader):
                x = data.to(self.device)

                output, f_x = network(x)
                d_f_x, _, _ = self.model_discriminator(f_x)
                z, _, _ = self.model_discriminator(x)

                loss, loss_bce, loss_reg_term = loss_function(output, target.to(self.device), d_f_x, z)

                all_loss += loss.item()
                batch_num += 1

                r2score = r2_score_function(d_f_x.T, z.T, multioutput='raw_values')
                r2_score_aggregator.extend(r2score.cpu().detach().numpy())

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss / batch_num)
        self.test_metrics.update(f'test/{log_prefix}_r2score', np.mean(r2_score_aggregator))

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)
