import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from data_loader.data_loaders import TextDatasetPairWrapper, MultiDataset, TextDataDataset
from logger import TensorboardWriter
from model.metric import get_label_classification_report
from model.model import TextClassifierModel, CrossDomainModel, CrossDomainModelPair
from model.loss import BinaryCrossEntropy, CrossDomainModelTrainLoss
from utils import MetricTracker


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


class TrainerMixedModels:
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, data_loader3=None, data_loader4=None,
                 lr_scheduler=None, len_epoch=None, **kwargs):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        if data_loader3 is not None:
            self.dataset3 = data_loader3.dataset
        if data_loader4 is not None:
            self.dataset4 = data_loader4.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'ABCD' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'ABCD' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'ABCD' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}_{i}/r2score' for c in 'ABCD' for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/loss_{i}{j}' for i in ['A', 'B', 'C', 'D']
              for j in ['A', 'B', 'C', 'D'] if i != j]
        m1 += [f'test/{i}{j}_{m}' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D']
               for m in get_label_classification_report() if i != j]
        m1 += [f'test/{i}{j}_r2score' for i in ['A', 'B', 'C', 'D']
               for j in ['A', 'B', 'C', 'D'] if i != j]
        self.test_metrics = MetricTracker(*m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0
        loss_function = BinaryCrossEntropy()

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        model1 = self.train_on_dataset(self.dataset1, batch_size_1, loss_function, log_prefix='A')

        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        model2 = self.train_on_dataset(self.dataset2, batch_size_2, loss_function, log_prefix='B')

        batch_size_3 = batch_size_2
        if self.dataset3 is not None:
            model3 = self.train_on_dataset(self.dataset3, batch_size_3,
                                           loss_function, log_prefix='C')

        batch_size_4 = self.config['data_loader_train4/args/batch_size'] or 16
        if self.dataset4 is not None:
            model4 = self.train_on_dataset(self.dataset4, batch_size_4,
                                           loss_function, log_prefix='D')

        # TEST
        self.test(self.dataset2, model1, batch_size_2, loss_function, log_prefix='AB')
        if self.dataset3 is not None:
            self.test(self.dataset3, model1, batch_size_3, loss_function, log_prefix='AC')
        if self.dataset4 is not None:
            self.test(self.dataset4, model1, batch_size_4, loss_function, log_prefix='AD')

        self.test(self.dataset1, model2, batch_size_1, loss_function, log_prefix='BA')
        if self.dataset3 is not None:
            self.test(self.dataset3, model2, batch_size_3, loss_function, log_prefix='BC')
        if self.dataset4 is not None:
            self.test(self.dataset4, model2, batch_size_4, loss_function, log_prefix='BD')

        if self.dataset3 is not None:
            self.test(self.dataset1, model3, batch_size_1, loss_function, log_prefix='CA')
            self.test(self.dataset2, model3, batch_size_2, loss_function, log_prefix='CB')
            if self.dataset4 is not None:
                self.test(self.dataset4, model3, batch_size_4, loss_function, log_prefix='CD')

        if self.dataset4 is not None:
            self.test(self.dataset1, model4, batch_size_1, loss_function, log_prefix='DA')
            self.test(self.dataset2, model4, batch_size_2, loss_function, log_prefix='DB')
            if self.dataset3 is not None:
                self.test(self.dataset3, model4, batch_size_3, loss_function, log_prefix='DC')

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
            network = TextClassifierModel(self.text_embedding_size).to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_metric = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

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
            current_loss = 0.0
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = network(inputs)

                loss = loss_function(outputs, targets)
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
                        f'{batch_idx / len_epoch:.0f}',
                        loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch',
                                      current_loss / batch_number)

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

        return current_loss / len(testloader)

    def test(self, dataset, network, batch_size, loss_function, log_prefix='A'):
        """
        Test on full dataset
        """
        self.logger.info(f'Process test dataset {log_prefix}')
        current_loss = .0
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

                current_loss += loss.item()
                batch_num += 1

                all_outputs.extend((output.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', current_loss / batch_num)

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerMultiMixedModels(TrainerMixedModels):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, data_loader3=None,
                 lr_scheduler=None, len_epoch=None, **kwargs):
        self.config = config
        self.device = device
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.text_embedding_size = self.config['text_embedding_size'] \
            if 'text_embedding_size' in self.config else 768

        self.metric_ftns = metric_ftns

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.k_fold = cfg_trainer['k_fold']
        self.save_period = cfg_trainer['save_period']

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.dataset3 = data_loader3.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}{j}_{i}/loss' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += [f'fold_{c}{j}_{i}/loss_epoch' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        m += [f'fold_{c}{j}_{i}/metrics/{m}' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)
              for m in get_label_classification_report()]
        m += [f'fold_{c}{j}_{i}/r2score' for c in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for i in range(self.k_fold)]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}{j}{k}_{m}' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for k in
              ['A', 'B', 'C', 'D'] for m in
              get_label_classification_report() if i != j]
        m1 += [f'test/{i}{j}{k}_r2score' for i in ['A', 'B', 'C', 'D'] for j in ['A', 'B', 'C', 'D'] for k in
              ['A', 'B', 'C', 'D'] if i != j]
        self.test_metrics = MetricTracker('test/loss_ABC', 'test/loss_ACB', 'test/loss_BCA',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        sample_perc = self.config['sample_perc']
        self.logger.info('Run with sample perc=' + str(sample_perc))
        self.training_step = 0

        loss_function = BinaryCrossEntropy()

        batch_size_1 = self.config['data_loader_train/args/batch_size'] or 16
        batch_size_2 = self.config['data_loader_valid/args/batch_size'] or 16
        batch_size_3 = self.config['data_loader_train3/args/batch_size'] or 16

        # TRAIN ON GOSSIP + POLITIFACT + FNKaggle, TEST on FNKaggle
        train3, test3 = train_test_split(self.dataset3.data, train_size=0.1, stratify=self.dataset3.data.label,
                                         random_state=123)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset3.data))
            train3, _ = train_test_split(train3, train_size=n, stratify=train3.label, random_state=123)

        model1 = self.train_on_dataset(MultiDataset(self.dataset1, self.dataset2, train3), batch_size_1, loss_function,
                                       log_prefix='AB')

        self.test(TextDataDataset(test3), model1, batch_size_3, loss_function, log_prefix='ABC')

        # TRAIN ON GOSSIP + FNKaggle + POLITIFACT, TEST on POLITIFACT
        train2, test2 = train_test_split(self.dataset2.data, train_size=0.1, stratify=self.dataset2.data.label,
                                         random_state=123)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset2.data))
            train2, _ = train_test_split(train2, train_size=n, stratify=train2.label, random_state=123)

        model2 = self.train_on_dataset(MultiDataset(self.dataset1, train2, self.dataset3), batch_size_3, loss_function,
                                       log_prefix='AC')

        self.test(TextDataDataset(test2), model2, batch_size_2, loss_function, log_prefix='ACB')

        # TRAIN ON POLITIFACT + FNKaggle + GOSSIP, TEST on GOSSIP
        train1, test1 = train_test_split(self.dataset1.data, train_size=0.1, stratify=self.dataset1.data.label)

        if sample_perc < 0.1:
            n = int(sample_perc * len(self.dataset1.data))
            train1, _ = train_test_split(train1, train_size=n, stratify=train1.label, random_state=123)

        model3 = self.train_on_dataset(MultiDataset(train1, self.dataset2, self.dataset3), batch_size_3, loss_function,
                                       log_prefix='BC')
        self.test(TextDataDataset(test1), model3, batch_size_1, loss_function, log_prefix='BCA')


class TrainerCRArch:
    """
    Trainer class
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
        self.dataset1 = data_loader.dataset
        self.dataset2 = valid_data_loader.dataset
        self.log_step = int(np.sqrt(data_loader.batch_size))

        metric_names = get_label_classification_report()
        print('USE ONLY', *metric_names)

        m = [f'fold_{c}_{i}/loss' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_classifier' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_discriminator' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_domain' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/loss_epoch' for c in 'AB' for i in range(self.k_fold)]
        m += [f'fold_{c}_{i}/metrics/{m}' for c in 'AB' for i in range(self.k_fold)
              for m in get_label_classification_report()]
        self.train_metrics = MetricTracker(*m, writer=self.writer)
        self.valid_metrics = MetricTracker(*m, writer=self.writer)

        m1 = [f'test/{i}_{m}' for i in 'AB' for m in get_label_classification_report()]
        self.test_metrics = MetricTracker('test/loss_A', 'test/loss_B',
                                          'test/loss_classifier_A', 'test/loss_classifier_B',
                                          'test/loss_discriminator_A', 'test/loss_discriminator_B',
                                          'test/loss_domain_A', 'test/loss_domain_B',
                                          *m1, writer=self.writer)

        self.training_step = 0

    def train(self):
        self.training_step = 0
        loss_function = CrossDomainModelTrainLoss()

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
            network = CrossDomainModel(p_dims=[64, 256, 768]).to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader) // trainloader.batch_size
        if len_epoch == 0:
            len_epoch = 1

        if getattr(loss_function, 'reset'):
            loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [.0, .0, .0, .0]
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = network(inputs)

                loss = loss_function(outputs, inputs, targets)
                loss.loss.backward()
                optimizer.step()

                current_loss[0] += loss.loss.item()
                current_loss[1] += loss.classifier_loss.item()
                current_loss[2] += loss.discriminator_loss.item()
                current_loss[3] += loss.domain_loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_classifier', loss.classifier_loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_discriminator', loss.discriminator_loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_domain', loss.domain_loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            if getattr(loss_function, 'update_alphas'):
                loss_function.update_alphas(current_loss[1] / batch_number,
                                            current_loss[2] / batch_number,
                                            current_loss[3] / batch_number)

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
        if getattr(loss_function, 'reset'):
            loss_function.reset()

        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, data, target.to(self.device))

                self.writer.set_step(self.training_step, 'valid')
                self.training_step += 1
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.loss.item())
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss_classifier', loss.classifier_loss.item())
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss_discriminator',
                                          loss.discriminator_loss.item())
                self.valid_metrics.update(f'fold_{log_prefix}_{fold}/loss_domain', loss.domain_loss.item())

                current_loss += loss.loss.item()

                all_outputs.extend((output.y_hat.detach().cpu().numpy() >= .5).astype(int))
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
        all_loss = [0, 0, 0, 0]
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        batch_num = 0
        network.eval()
        if getattr(loss_function, 'reset'):
            loss_function.reset()

        # self.test_metrics.reset()
        with torch.no_grad():
            all_outputs = []
            all_targets = []

            for batch_idx, (data, target) in enumerate(testloader):
                data = data.to(self.device)

                output = network(data)
                loss = loss_function(output, data, target.to(self.device))

                all_loss[0] += loss.loss.item()
                all_loss[1] += loss.classifier_loss.item()
                all_loss[2] += loss.discriminator_loss.item()
                all_loss[3] += loss.domain_loss.item()
                batch_num += 1

                all_outputs.extend((output.y_hat.detach().cpu().numpy() >= .5).astype(int))
                all_targets.extend(target.numpy().astype(int))

            all_outputs = np.array(all_outputs)
            all_targets = np.array(all_targets)

        self.writer.set_step(self.train_metrics.count_steps(), 'valid')
        self.test_metrics.process_metrics(self.metric_ftns, f'test/{log_prefix}_', all_outputs, all_targets)
        self.test_metrics.update(f'test/loss_{log_prefix}', all_loss[0] / batch_num)
        self.test_metrics.update(f'test/loss_classifier_{log_prefix}', all_loss[1] / batch_num)
        self.test_metrics.update(f'test/loss_discriminator_{log_prefix}', all_loss[2] / batch_num)
        self.test_metrics.update(f'test/loss_domain_{log_prefix}', all_loss[3] / batch_num)

        log = self.test_metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        filename = str(self.checkpoint_dir / f'checkpoint-model-{log_prefix}.pth')
        save_model(network, filename)


class TrainerCRArchPairTraining(TrainerCRArch):
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
                TextDatasetPairWrapper(dataset), batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            network = CrossDomainModelPair(p_dims=[64, 256, 768]).to(self.device)
            network.apply(reset_weights)

            self._train_on_fold(fold, loss_function, network, num_epochs, trainloader, log_prefix=log_prefix)
            valid_loss = self._test_on_fold(fold, loss_function, network, testloader, log_prefix=log_prefix)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = network

        return best_model

    def _train_on_fold(self, fold, loss_function, network, num_epochs, trainloader, log_prefix='A'):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        len_epoch = len(trainloader) // trainloader.batch_size
        if len_epoch == 0:
            len_epoch = 1

        if getattr(loss_function, 'reset'):
            loss_function.reset()

        for epoch in range(0, num_epochs):
            current_loss = [.0, .0, .0, .0]
            batch_number = 0

            # Iterate over the DataLoader for training data
            for batch_idx, data in enumerate(trainloader, 0):

                inputs, targets, x2, y2 = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                x2, y2 = x2.to(self.device), y2.to(self.device)
                optimizer.zero_grad()
                outputs = network(inputs, x2)

                loss = loss_function(outputs, inputs, targets, y2, epoch=epoch)
                loss.loss.backward()
                optimizer.step()

                current_loss[0] += loss.loss.item()
                current_loss[1] += loss.classifier_loss.item()
                current_loss[2] += loss.discriminator_loss.item()
                current_loss[3] += loss.domain_loss.item()

                self.writer.set_step(self.training_step)
                self.training_step += 1
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss', loss.loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_classifier', loss.classifier_loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_discriminator', loss.discriminator_loss.item())
                self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_domain', loss.domain_loss.item())
                batch_number += 1

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch Fold-{}: {} ({}%) Loss: {:.6f}'.format(
                        fold,
                        epoch,
                        f'{batch_idx / len_epoch:.0f}',
                        loss.loss.item()))

            self.train_metrics.update(f'fold_{log_prefix}_{fold}/loss_epoch', current_loss[0] / batch_number)
            if getattr(loss_function, 'update_alphas'):
                loss_function.update_alphas(current_loss[1] / batch_number,
                                            current_loss[2] / batch_number,
                                            current_loss[3] / batch_number)

            log = self.train_metrics.result()
            log['epoch'] = epoch
            for key, value in log.items():
                if key.startswith(f'fold_{log_prefix}_{fold}'):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

        self.logger.info('Training process has finished. Saving trained model.')

