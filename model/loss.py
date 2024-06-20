from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, defaultdict

from torch import Tensor
from torcheval.metrics.functional import r2_score as r2_score_function

from model.model import MoEResult, MoEResultWithDomain

CrossDomainModelTrainLossResult = namedtuple('CrossDomainModelTrainLossResult',
                                             'loss, classifier_loss, discriminator_loss, domain_loss')


def nll_loss(output, target):
    return F.nll_loss(output, target)


def BCELoss_ClassWeights(inputT, target, class_weights):
    """
    BCE loss with class weights
    :param inputT: (n, d)
    :param target: (n, d)
    :param class_weights: (d,)
    :return:
    """
    inputT = torch.clamp(inputT, min=1e-7, max=1 - 1e-7)
    w0, w1 = class_weights
    bce = - w0 * target * torch.log(inputT) - w1 * (1 - target) * torch.log(1 - inputT)
    # weighted_bce = (bce * class_weights).sum(axis=1) / class_weights.sum(axis=1)[0]
    # final_reduced_over_batch = weighted_bce.mean(axis=0)
    final_reduced_over_batch = bce.mean(axis=0)
    return final_reduced_over_batch


class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            self.bce = torch.nn.BCELoss()
        else:
            self.bce = partial(BCELoss_ClassWeights, class_weights=class_weights)

    def forward(self, y_pred, y_true):
        return self.bce(y_pred.flatten(), y_true.flatten())

    def reset(self):
        pass


class AutoEncoderGeneric(nn.Module):
    """
    MSE loss
    Takes embeddings generate and target
    """
    def __init__(self, main_loss=torch.nn.CrossEntropyLoss(), is_vae_model=False):
        super().__init__()
        self._mainLoss = main_loss
        self._is_vae_model = is_vae_model

    def forward(self, y, target, mu=None, log_var=None):
        if not self._is_vae_model:
            return self._mainLoss(y, target)

        kld = self.kld(mu, log_var)

        rec_error = self._mainLoss(y, target)

        return rec_error + kld

    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return kld


class AutoEncoderMSE(nn.Module):
    """
    MSE loss
    Takes embeddings generate and target
    """
    def __init__(self, is_vae_model=False, is_classifier_model=False):
        super().__init__()
        self._mse = torch.nn.MSELoss()
        self._is_vae_model = is_vae_model
        self._is_classifier_model = is_classifier_model
        self.bce = BinaryCrossEntropy()
        self.alphas = DynamicWeightAverage(3)

    def reset(self):
        self.alphas.reset()

    def update_alphas(self, reconstruction_error, kld_error, classifier_error):
        self.alphas.update(0, reconstruction_error)
        self.alphas.update(1, kld_error)
        self.alphas.update(2, classifier_error)

    def forward(self, x_rec, x, mu=None, log_var=None, y_pred=None, y=None):
        assert len(x.shape) == 2 and x.shape == x_rec.shape, f'Bad dimension {x_rec.shape} != {x.shape}'
        if not self._is_vae_model:
            return self._mse(x_rec, x)

        kld = self.kld(mu, log_var)

        rec_error = self._mse(x_rec, x)

        if self._is_classifier_model:
            # alphas = self.alphas.get_values()
            bce = self.bce(y_pred, y)
            # return alphas[0] * rec_error + alphas[1] * kld + alphas[2] * bce
            return rec_error + kld + 2 * bce

        return rec_error + kld

    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return kld


class AutoEncodeMultiMode(nn.Module):
    def __init__(self, is_vae_model=False):
        super().__init__()
        self.mseLoss = AutoEncoderMSE(is_vae_model)
        self.pairLoss = nn.CosineEmbeddingLoss()

    def forward(self, y, target, mu=None, log_var=None, y2=None, target2=None, epoch=None):
        loss = self.mseLoss(y, target, mu, log_var)
        if epoch and epoch >= 3:
            loss += self.pairLoss(y, y2, target2)

        return loss


class AutoEncoderAdv(nn.Module):
    def __init__(self, is_vae_model=False):
        super().__init__()
        self.mseLossAE = AutoEncoderMSE(is_vae_model)
        self.mse = nn.MSELoss()

    def forward(self, y, target, mu=None, log_var=None, y2=None, target2=None, epoch=None):
        loss = self.mseLossAE(y, target, mu, log_var)

        loss += self.mse(y2, target2)

        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_hat, z):
        """
        Compute the loss for classification module
        :param z_hat: bs x emb_size
        :param z: bs x emb_size
        :return:
        """
        l_d = torch.linalg.vector_norm(z_hat - z, ord=2, dim=1).mean()

        return - l_d * 1e-4


class CrossDomainModelTrainLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_classifier = BinaryCrossEntropy()
        self.loss_discriminator = DiscriminatorLoss()
        # self.loss_domain = AutoEncoderMSE(is_vae_model=True)
        self.loss_domain = AutoEncodeMultiMode(is_vae_model=True)
        self.alphas = DynamicWeightAverage(3)

    def reset(self):
        self.alphas.reset()

    def update_alphas(self, classifier_loss, discriminator_loss, domain_loss):
        self.alphas.update(0, classifier_loss)
        self.alphas.update(1, discriminator_loss)
        self.alphas.update(2, domain_loss)

    def forward(self, y_output, x, y, y2=None, epoch=0):
        """

        :param y_output: output of CrossDomainModel model
        :param x: bs x text_emb, text embedding
        :param y: bs, target label
        :return: loss components as named tuple
        """
        classifier_loss = self.loss_classifier(y_output.y_hat, y)
        discriminator_loss = self.loss_discriminator(y_output.z_hat, y_output.mu.detach())
        domain_loss = self.loss_domain(y_output.z_domain, x, y_output.mu, y_output.log_var,
                                       y_output.z_domain2, y2, epoch=epoch)

        # alpha = 1
        # beta = 1e-2
        # gamma = 10
        # loss = alpha * classifier_loss \
        #     - beta * discriminator_loss \
        #     + gamma * domain_loss

        loss = classifier_loss + discriminator_loss + domain_loss
        # alphas = self.alphas.get_values()
        # loss = alphas[0] * classifier_loss \
        #     + alphas[1] * discriminator_loss \
        #     + alphas[2] * domain_loss

        return CrossDomainModelTrainLossResult(loss, classifier_loss, discriminator_loss, domain_loss)


class CrossDomainModelAutoEncoderBasedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()

        self.bce = BinaryCrossEntropy(class_weights=class_weights)
        self.mse = nn.MSELoss(reduction='sum')
        self.alphas = DynamicWeightAverage(2)

    def reset(self):
        self.alphas.reset()

    def update_alphas(self, classifier_loss, discriminator_loss):
        self.alphas.update(0, classifier_loss)
        self.alphas.update(1, discriminator_loss)

    def forward(self, y, targets, d_f_x, z):
        """

        :param y_output: output of CrossDomainModel model
        :param x: bs x text_emb, text embedding
        :param y: bs, target label
        :return: loss components as named tuple
        """
        assert len(z.shape) == 2 and z.shape == d_f_x.shape, f'fail z shape {z.shape} != {d_f_x.shape}'

        bce = self.bce(y, targets)
        # mse = self.mse(d_f_x, z)

        # embed dim size
        # bs, k = z.shape
        # log_term = torch.log(4 * k * bs - mse)
        # loss = bce + .01 * log_term

        # alphas = self.alphas.get_values()
        # log_term = torch.nn.functional.cosine_similarity(d_f_x, z, dim=1, eps=1e-8).mean()
        # loss = alphas[0] * bce + alphas[1] * log_term

        # k = z.shape[1]
        # log_term = torch.log(4 * k - mse)
        # loss = bce + log_term

        log_term = torch.exp(-(d_f_x - z).abs().sum(dim=1)).mean()
        # distance = torch.nn.functional.cosine_similarity(d_f_x, z, dim=1, eps=1e-8)
        # log_term = torch.exp(-distance).mean()
        loss = bce + log_term

        return loss, bce, log_term


class CrossDomainModelAutoEncoderWithFLLoss(nn.Module):
    def __init__(self, gamma=2.):
        super().__init__()

        self.classification_loss = FocalLoss(gamma=gamma)
        self.mse = nn.MSELoss(reduction='sum')
        self.alphas = DynamicWeightAverage(2)

    def reset(self):
        self.alphas.reset()

    def update_alphas(self, *args):
        for i, term in enumerate(args):
            self.alphas.update(i, term)

    def forward(self, y, targets, d_f_x, z):
        assert len(z.shape) == 2 and z.shape == d_f_x.shape, \
            f'fail z shape {z.shape} != {d_f_x.shape}'

        term1 = self.classification_loss(y, targets)
        mse = self.mse(d_f_x, z)

        # embed dim size
        bs, k = z.shape

        log_term = torch.log(4 * k * bs - mse)

        # alphas = self.alphas.get_values()
        # loss = alphas[0] * term1 + alphas[1] * log_term
        loss = term1 + .01 * log_term

        return loss, term1, log_term


class CrossDomainModelAutoEncoderBasedLossR2(CrossDomainModelAutoEncoderBasedLoss):
    def forward(self, y, targets, d_f_x, z):
        """

        :param y_output: output of CrossDomainModel model
        :param x: bs x text_emb, text embedding
        :param y: bs, target label
        :return: loss components as named tuple
        """
        assert len(z.shape) == 2 and z.shape == d_f_x.shape, f'fail z shape {z.shape} != {d_f_x.shape}'

        bce = self.bce(y, targets)

        # r2 = torch.mean(r2_score_function(d_f_x.T, z.T, multioutput='raw_values'))
        # log_term = - torch.log(1 - torch.abs(r2))

        r2 = torch.mean(r2_score_function(d_f_x.T, z.T, multioutput='raw_values'))
        log_term = torch.log1p(torch.abs(r2))

        loss = bce + .1 * log_term

        return loss, bce, log_term


class TextClassifierModelTrainLoss(nn.Module):
    def __init__(self, alpha_param=.5):
        super().__init__()
        self.loss_classifier = BinaryCrossEntropy()
        self.alpha_param = alpha_param
        self.alphas = DynamicWeightAverage(2)

    def reset(self):
        self.alphas.reset()

    def update_alphas(self, classifier_loss, domain_loss):
        self.alphas.update(0, classifier_loss)
        self.alphas.update(1, domain_loss)

    def forward(self, y_hat, y, d_fx=None):
        """

        :param y_hat: output of classifier model
        :param y: bs, target label
        :param d_fx: discriminator
        :return: loss
        """
        classifier_loss = self.loss_classifier(y_hat, y)

        if d_fx is not None:
            loss_dfx = torch.log1p(torch.abs(self.alpha_param - d_fx)).mean()

            alphas = self.alphas.get_values()
            loss = alphas[0] * classifier_loss + alphas[1] * loss_dfx
            return loss, classifier_loss, loss_dfx
        else:
            return classifier_loss, 0, 0


class DynamicWeightAverage(nn.Module):
    """
    Compute weights for the loss components
    See Liu, Shikun, Edward Johns, and Andrew J. Davison. "End-to-end multi-task learning with attention."
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

    For K component:
    alpha(k) = K * softmax(w(t-1) / T)
    w_k(t-1) = Loss_k(t-1) / Loss_k(t-2)
    w_k(1) = w_k(2) = 1
    """
    def __init__(self, kappa_param=1, temperature=1):
        assert kappa_param > 1, 'K must be greater than 1'
        assert temperature >= 1, 'T must be greater or equals than 1'
        super().__init__()
        self.kappa = kappa_param
        self.temperature = temperature
        self.history = defaultdict(list)
        self.iteration = 0

    def reset(self):
        self.iteration = 0
        self.history.clear()

    def update(self, k, loss_value):
        """
        Update the loss value for each epoch
        :param k: component id
        :param loss_value: average epoch loss
        :return:
        """
        assert 0 <= k < self.kappa, f'kappa must be in [0, {self.kappa}]. Given {k}'
        # assert isinstance(loss_value, torch.Tensor), 'loss value must be a tensor'

        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        previous_value = self.history[k]

        if len(previous_value) == 2:
            del previous_value[0]

        previous_value.append(loss_value)

    def get_values(self):
        alphas = torch.ones(self.kappa)
        for i in range(self.kappa):
            alphas[i] = self._compute_alpha(i)

        alphas = self.kappa * torch.softmax(alphas / self.temperature, dim=0)

        return alphas

    def _compute_alpha(self, k):
        previous_value = self.history[k]

        if len(previous_value) != 2:
            # i.e. epoch iterations <= 2
            return torch.tensor(1, dtype=torch.float32)

        loss_prev = torch.tensor(previous_value[1], dtype=torch.float32)
        loss_prev_prev = torch.tensor(previous_value[0], dtype=torch.float32)
        return loss_prev / loss_prev_prev


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        """
        Compute the sum of: - alpha * (1-pt)**gamma * log(pt)
        with pt = is the probability, i.e. output of the sigmoid
        :param alpha:
        :param gamma:
        :param reduction:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_hat, target):
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        assert len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[1] == 1)

        y_hat = y_hat.flatten()  # for bce
        target = target.flatten()  # for bce

        # self.alpha = torch.full_like(y_hat, 1)
        # self.alpha[target == 0] = 10
        # self.alpha = torch.full_like(y_hat, 10)
        # self.alpha[target == 0] = 1

        # ce_loss = F.cross_entropy(y_hat, target, weight=self.alpha, reduction='none')
        ce_loss = F.binary_cross_entropy(y_hat, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


## LOSS FOR MOE MODELS
LossResult = namedtuple('LossResult', ['loss', 'bce_loss', 'l1_loss', 'balancing_loss'])
LossResultWithDomain = namedtuple('LossResultWithDomain',
                                  ['loss', 'bce_loss', 'l1_loss', 'balancing_loss', 'domain_loss'])


def BCELossLike_MoE(output, true, weights, class_weights):
    """
    BCE loss with class weights
    :param inputT: (n, d)
    :param target: (n, d)
    :param class_weights: (d,)
    :return:
    """
    if len(true.shape) == 1:
        true = true.reshape(-1, 1)
    batch_size = output.shape[0]
    class_weights_batch = torch.full((batch_size,1), class_weights[0], device=output.device)
    class_weights_batch[true == 1] = class_weights[1]
    #class_weights_batch = class_weights_batch.squeeze()
    true = 2 * true - 1
    output_norm = torch.sigmoid(output * true.unsqueeze(1))
    loss = - torch.log( 1/output.shape[1] * torch.bmm( weights.unsqueeze(1), output_norm ) ) * class_weights_batch #/ (class_weights[0] + class_weights[1])
    loss = torch.mean(loss)
    return loss


def LoadBalancingLoss(gate_logits, lambda_balancing=0.0001):
    # Compute the average usage of each expert
    expert_usage = gate_logits.mean(dim=0)
    # Penalize deviation from uniform usage
    loss = torch.std(expert_usage)

    return lambda_balancing * loss


def L1Loss(model, lambda_l1_exp=0.005, lambda_l1_gate=0.1):
    l1_norm_exp = sum(p.abs().sum() for p in model.experts.parameters())
    l1_norm_gate = sum(p.abs().sum() for p in model.gate.parameters())

    return lambda_l1_exp * l1_norm_exp + lambda_l1_gate * l1_norm_gate


class MoELoss(torch.nn.Module):
    def __init__(self, lambda_l1_gate, lambda_l1_exp, lambda_balancing, class_weights=(1, 1)):
        super().__init__()
        self.lambda_balancing = lambda_balancing
        self.lambda_l1_exp = lambda_l1_exp
        self.lambda_l1_gate = lambda_l1_gate
        # self.bce = partial(BCELossLike_MoE, class_weights=class_weights)
        self.bce = BinaryCrossEntropy(class_weights=class_weights)

    def forward(self, moe_outputs: MoEResult, targets: Tensor, model, validation=False):
        # bce_loss = self.bce(moe_outputs.experts_outputs, targets, moe_outputs.gate_probs)
        bce_loss = self.bce(moe_outputs.weighted_outputs, targets)
        if self.lambda_balancing != 0:
            balancing_loss = LoadBalancingLoss(moe_outputs.gate_probs, self.lambda_balancing)
        else:
            balancing_loss = torch.tensor([0], device=moe_outputs.experts_outputs.device)
        if (self.lambda_l1_exp !=0 or self.lambda_l1_gate != 0) and validation is False:
            l1_loss = L1Loss(model, self.lambda_l1_exp, self.lambda_l1_gate)
        else:
            l1_loss = torch.tensor([0], device=moe_outputs.experts_outputs.device)

        loss = bce_loss + balancing_loss + l1_loss

        return LossResult(loss, bce_loss, l1_loss, balancing_loss)


class MoELossWithDomain(torch.nn.Module):
    def __init__(self, lambda_l1_gate, lambda_l1_exp, lambda_balancing, class_weights=(1, 1)):
        super().__init__()
        self.lambda_balancing = lambda_balancing
        self.lambda_l1_exp = lambda_l1_exp
        self.lambda_l1_gate = lambda_l1_gate
        # self.bce = partial(BCELossLike_MoE, class_weights=class_weights)
        self.bce = BinaryCrossEntropy(class_weights=class_weights)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, moe_outputs: MoEResultWithDomain, targets: Tensor, srcs: Tensor,
                model, validation=False):
        # bce_loss = self.bce(moe_outputs.experts_outputs, targets, moe_outputs.gate_probs)
        bce_loss = self.bce(moe_outputs.weighted_outputs, targets)
        if self.lambda_balancing != 0:
            balancing_loss = LoadBalancingLoss(moe_outputs.gate_probs, self.lambda_balancing)
        else:
            balancing_loss = torch.tensor([0], device=moe_outputs.experts_outputs.device)
        if (self.lambda_l1_exp !=0 or self.lambda_l1_gate != 0) and validation is False:
            l1_loss = L1Loss(model, self.lambda_l1_exp, self.lambda_l1_gate)
        else:
            l1_loss = torch.tensor([0], device=moe_outputs.experts_outputs.device)

        if not validation:
            domain_loss = self.ce(moe_outputs.domain_outputs, srcs)
        else:
            domain_loss = torch.tensor([0], device=moe_outputs.experts_outputs.device)

        loss = bce_loss + .1 * domain_loss + balancing_loss + l1_loss

        return LossResultWithDomain(loss, bce_loss, l1_loss, balancing_loss, domain_loss)


class MoEBCELoss(torch.nn.Module):
    def __init__(self, class_weights=(1, 1)):
        super().__init__()
        self.bce = BinaryCrossEntropy(class_weights=class_weights)

    def forward(self, moe_outputs: MoEResult, targets: Tensor, model, validation=False):

        bce_loss = self.bce(moe_outputs.weighted_outputs, targets)
        loss = bce_loss
        balancing_loss, l1_loss = torch.tensor([0]), torch.tensor([0])

        return LossResult(loss, bce_loss, l1_loss, balancing_loss)
