import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from collections import namedtuple
from model.sparse_moe import SparseDispatcher, MoE as SparseMoE


CrossDomainModelResult = namedtuple('CrossDomainModelResult',
                                    'z_domain, mu, log_var, y_hat, f_x, z_hat, z_domain2')

class BaseClassifier(nn.Module):
    def __init__(self, input_dim = 768):
        super(BaseClassifier, self).__init__()

        self._net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.Tanh(),
            nn.Linear(input_dim // 8, 1)
        )

    def forward(self, x):
        x1 = self._net(x)
        return x1

class TextClassifierModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._net(x)


class TextClassifierDebugModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._net(x)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value you are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class MultiVAE(BaseModel):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=None,
                 use_enc_bn=False, use_dec_bn=False, **k_args):
        super().__init__()
        self.p_dims = p_dims
        self.use_enc_bn = use_enc_bn
        self.use_dec_bn = use_dec_bn
        if q_dims:
            # assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        print('Encoder dimensions:', self.q_dims)
        print('Decoder dimensions:', self.p_dims)

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        if self.use_enc_bn:
            self.bn_enc = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in temp_q_dims[1:-1]])
        if self.use_dec_bn:
            self.bn_dec = nn.ModuleList([nn.BatchNorm1d(d_out) for d_out in self.p_dims[1:-1]])

        self.drop = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        if not self.training:
            return self.decode(mu), mu, log_var

        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def encode(self, input_data):
        # h = F.normalize(input_data)
        # h = self.drop(h)
        h = input_data
        mu, log_var = None, None
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                if self.use_enc_bn:
                    h = self.bn_enc[i](h)
                h = torch.tanh(h)
                # h = torch.relu(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                log_var = h[:, self.q_dims[-1]:]

        return mu, log_var

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add_(mu)

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                if self.use_dec_bn:
                    h = self.bn_dec[i](h)
                h = torch.tanh(h)
                # h = torch.relu(h)
        return h


class NoisedVAE(MultiVAE):
    def __init__(self, **k_args):
        super().__init__(**k_args)
        self.noise_level = GaussianNoise(.05)

    def forward(self, input_data):
        x = self.noise_level(input_data)
        return super().forward(x)


class SimpleAE(BaseModel):
    def __init__(self, p_dims, q_dims=None, use_dropout=False):
        super().__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        print('Encoder dimensions:', self.q_dims)
        print('Decoder dimensions:', self.p_dims)

        # Last dimension of q- network is for z
        q_list, p_list = [], []

        for i, d_in, d_out in zip(range(len(self.q_dims) - 1), self.q_dims[:-1], self.q_dims[1:]):
            q_list.append(nn.Linear(d_in, d_out))
            if i < len(self.q_dims) - 1:
                q_list.append(nn.Tanh())
            if use_dropout and i < len(self.q_dims) - 2:
                q_list.append(nn.Dropout(p=.25))

        for i, d_in, d_out in zip(range(len(self.p_dims) - 1), self.p_dims[:-1], self.p_dims[1:]):
            p_list.append(nn.Linear(d_in, d_out))
            if i < len(self.p_dims) - 1:
                p_list.append(nn.Tanh())
            if use_dropout and i < len(self.p_dims) - 2:
                p_list.append(nn.Dropout(p=.25))

        self.q_layers = nn.ModuleList(q_list)
        self.p_layers = nn.ModuleList(p_list)

    def forward(self, input_data):
        z = self.encode(input_data)
        return self.decode(z), z

    def encode(self, input_data):
        h = input_data
        for layer in self.q_layers:
            h = layer(h)
        return h

    def decode(self, z):
        h = z
        for layer in self.p_layers:
            h = layer(h)
        return h


class DomainAutoencoderAndMapper(nn.Module):
    def __init__(self, input_size=768, use_bn_mapper=False, **k_args):
        super().__init__()

        if use_bn_mapper:
            self.mapper = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.BatchNorm1d(input_size),
                nn.Tanh()
            )
        else:
            self.mapper = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Tanh()
            )
        # self.mapper = lambda x: x
        self.ae = NoisedVAE(**k_args)

    def forward(self, x, use_mapper=True, use_output_activation=True):
        if use_mapper:
            x = self.mapper(x)

        z_domain, mu, log_var = self.ae(x)

        if use_output_activation:
            z_domain = self.mapper(z_domain)

        return z_domain, mu, log_var

    def get_mapping(self, x):
        return self.mapper(x)


class DomainAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ae = NoisedVAE(**kwargs)

    def forward(self, x, *args, **kwargs):
        z_domain, mu, log_var = self.ae(x)
        return z_domain, mu, log_var

    def get_mapping(self, x):
        # legacy
        return x


class DomainMultiTargetAndMapper(nn.Module):
    def __init__(self, input_size=768, final_activation=nn.Softmax(dim=1), **k_args):
        super().__init__()

        self.mapper = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh()
        )
        self.ae = MultiVAE(**k_args)
        self.final_activation = final_activation

    def forward(self, x, use_mapper=True):
        if use_mapper:
            x = self.mapper(x)

        z_domain, mu, log_var = self.ae(x)

        if self.final_activation is not None:
            z_domain = self.final_activation(z_domain)

        return z_domain, mu, log_var

    def encode(self, x, use_mapper=True):
        if use_mapper:
            x = self.mapper(x)

        _, mu, _ = self.ae(x)

        return mu

    def get_mapping(self, x):
        return self.mapper(x)

class DomainMultiTargetAndMapperPluggable(nn.Module):
    """
    See replace_decoder()
    """
    def __init__(self, input_size=768, final_activation=nn.Softmax(dim=1), **k_args):
        super().__init__()

        self.mapper = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh()
        )
        self.ae = MultiVAE(**k_args)
        self.final_activation = final_activation

    def replace_decoder(self, p_dims):
        assert p_dims[0] == self.ae.q_dims[-1], f'incompatible dimensions {self.ae.q_dims} <> {p_dims}'
        self.ae.p_dims = p_dims
        self.ae.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(p_dims[:-1], p_dims[1:])])

        for param in itertools.chain.from_iterable((self.mapper.parameters(), self.ae.q_layers.parameters())):
            param.requires_grad = False

        self.final_activation = None

    def forward(self, x, use_mapper=True):
        if use_mapper:
            x = self.mapper(x)

        z_domain, mu, log_var = self.ae(x)

        if self.final_activation is not None:
            z_domain = self.final_activation(z_domain)

        return z_domain, mu, log_var

    def get_mapping(self, x):
        return self.mapper(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, e_x):
        return self.net(e_x)


class DomainClassifierLarge(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.Tanh(),
            nn.Linear(input_size * 2, input_size // 4),
            nn.Tanh(),
            nn.Linear(input_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, e_x):
        return self.net(e_x)


class FakeNewsClassifier(BaseModel):
    def __init__(self, f_x_layers=(768, 256, 64), g_layers=(64, 1), use_dropout=False, **k_args):
        assert f_x_layers and len(f_x_layers) > 1
        assert g_layers and len(g_layers) > 1
        assert f_x_layers[-1] == g_layers[0], 'f_x output must be equals to G/D input'
        assert g_layers[-1] == 1, 'D output must be 1'
        super().__init__()

        net1 = []
        for i, d_in, d_out in zip(range(len(f_x_layers) - 1), f_x_layers[:-1], f_x_layers[1:]):
            net1.append(nn.Linear(d_in, d_out))

            if i < len(f_x_layers) - 2:
                net1.append(nn.Tanh())

            if use_dropout and i < len(f_x_layers) - 2:
                net1.append(nn.Dropout(p=.25))


        self.f_x = nn.Sequential(*net1)

        net2 = [(nn.Linear(d_in, d_out), nn.Tanh()) for
                d_in, d_out in zip(g_layers[:-1], g_layers[1:])]
        net2[-1] = (net2[-1][0], nn.Sigmoid())  # last activation is the classification
        self.G = nn.Sequential(*[item for sublist in net2 for item in sublist])

    def forward(self, x):
        f_x = self.f_x(x)
        y = self.G(f_x)
        return y, f_x


class FakeNewsDiscriminator(BaseModel):
    def __init__(self, d_layers=(64, 48, 64), **k_args):
        assert d_layers and len(d_layers) > 1
        super().__init__()

        net3 = [(nn.Linear(d_in, d_out), nn.Tanh()) for
                d_in, d_out in zip(d_layers[:-1], d_layers[1:])]
        net3[-1] = (net3[-1][0],)  # remove last activation
        self.D = nn.Sequential(*[item for sublist in net3 for item in sublist])

    def forward(self, f_x):
        z = self.D(f_x)
        return z


class CrossDomainModel(nn.Module):
    def __init__(self, **k_args):
        super().__init__()

        self.classifier = FakeNewsClassifier(**k_args)
        self.discriminator = FakeNewsDiscriminator(**k_args)
        self.domain = NoisedVAE(**k_args)

    def forward(self, x):
        z_domain, mu, log_var = self.domain(x)
        y_hat, f_x = self.classifier(x)
        z_hat = self.discriminator(f_x)

        return CrossDomainModelResult(z_domain, mu, log_var, y_hat, f_x, z_hat, None)


class CrossDomainModelPair(nn.Module):
    def __init__(self, **k_args):
        super().__init__()

        self.classifier = FakeNewsClassifier(**k_args)
        self.discriminator = FakeNewsDiscriminator(**k_args)
        self.domain = NoisedVAE(**k_args)

    def forward(self, x, x2=None):
        z_domain, mu, log_var = self.domain(x)
        y_hat, f_x = self.classifier(x)
        z_hat = self.discriminator(f_x)

        if x2 is not None:
            z_domain2, _, _ = self.domain(x2)
        else:
            z_domain2 = None

        return CrossDomainModelResult(z_domain, mu, log_var, y_hat, f_x, z_hat, z_domain2)


class CrossDomainAutoEncoderModel(nn.Module):
    def __init__(self, f_x_layers=(768, 256, 768), g_layers=(768, 256, 1), use_dropout=False,
                 p_rate=0.5, **k_args):
        assert f_x_layers and len(f_x_layers) > 1
        assert g_layers and len(g_layers) > 1
        # assert f_x_layers[-1] == g_layers[0], 'f_x output must be equals to G/D input'
        assert g_layers[-1] == 1, 'D output must be 1'
        super().__init__()

        net1 = []
        for i, d_in, d_out in zip(range(len(f_x_layers) - 1), f_x_layers[:-1], f_x_layers[1:]):
            net1.append(nn.Linear(d_in, d_out))

            # if i < len(f_x_layers) - 2:
            #     net1.append(nn.LeakyReLU())
            net1.append(nn.Tanh())

            if use_dropout and i < len(f_x_layers) - 2:
                # net1.append(GaussianNoise(sigma=.1))
                net1.append(nn.Dropout(p=p_rate))

        self.f_x = nn.Sequential(*net1)

        # net2 = [(nn.Linear(g_layers[0]+f_x_layers[-1], g_layers[0]), nn.Tanh())]
        net2 = [(nn.Linear(d_in, d_out), nn.Tanh()) for
                d_in, d_out in zip(g_layers[:-1], g_layers[1:])]
        net2[-1] = (net2[-1][0], nn.Sigmoid())  # last activation is the classification
        self.G = nn.Sequential(*[item for sublist in net2 for item in sublist])

        self.dropout = nn.Dropout(p=p_rate)

    def forward(self, x):
        f_x = self.f_x(x)
        # x_cat = torch.cat((x, f_x), dim=1)
        # x_cat = torch.add(x, f_x)
        # x_cat = self.dropout(x_cat)
        y = self.G(f_x)

        return y, f_x


class TextClassifierWithFeatureExtractionModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()

        self._fx = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, input_dim),
        )

        self._net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 8),
            nn.Tanh(),
            nn.Linear(input_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_x = self._fx(x)
        return self._net(f_x), f_x



class MoeExpert(nn.Module):
    def __init__(self, input_dim=768, model_file=None,
                 use_hard_prediction=False,
                 trainable=False,
                 activation=nn.Sigmoid()):
        """
        Create a freezed base model for moe.
        :param input_dim: dimension of input data
        :param model_file: path to model dump. This model will be freezed
        :param activation: activation function or None
        """
        super().__init__()
        self.classifier = BaseClassifier(input_dim)
        self.use_hard_prediction = use_hard_prediction
        self.activation = activation
        if model_file:
            self.classifier.load_state_dict(torch.load(model_file))
            for p in self.classifier.parameters():
                p.requires_grad = False

        if trainable:
            if use_hard_prediction:
                self.use_hard_prediction = False
                print('Using hard prediction is not possible with trainable parameters')
            last_level = self.classifier._net[-1]
            self.classifier._net[-1] = nn.Sequential(
                nn.Linear(last_level.in_features, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )

    def forward(self, x):
        y = self.classifier(x)
        if self.activation is not None:
            y = self.activation(y)
            if self.use_hard_prediction:
                y = (y > .5).float()
        return y


MoEResult = namedtuple('MoEResult',
                       ['weighted_outputs', 'experts_outputs', 'gate_probs'])
MoEResultWithDomain = namedtuple('MoEResultWithDomain',
                       ['weighted_outputs', 'experts_outputs',
                        'gate_probs', 'domain_outputs'])
MoEResultWithLoss = namedtuple('MoEResultWithLoss',
                       ['weighted_outputs', 'experts_outputs', 'gate_probs', 'loss'])


class MoE(BaseModel):
    def __init__(self, input_size, expert_model_dump,
                 use_hard_prediction=False, trainable_expert=False):
        super().__init__()
        assert isinstance(input_size, int), 'input_size must be an integer'
        assert isinstance(expert_model_dump, (list, tuple)), 'expert_model_dump must be an list'
        self.num_experts = len(expert_model_dump)
        self.experts = nn.ModuleList([MoeExpert(input_size, model_dump,
                                                use_hard_prediction, trainable=trainable_expert)
                                      for model_dump in expert_model_dump])
        self.gate = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts)
        )

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=1)
        #gate_probs = F.gumbel_softmax(gate_scores, tau=self.temperature, hard=False, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_outputs = torch.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)

        return MoEResult(weighted_outputs, expert_outputs, gate_probs)


class MoEWithDomainDetector(BaseModel):
    def __init__(self, input_size, expert_model_dump, domain_size=1,
                 use_hard_prediction=False, trainable_expert=False):
        super().__init__()
        assert isinstance(input_size, int), 'input_size must be an integer'
        assert isinstance(expert_model_dump, (list, tuple)), 'expert_model_dump must be an list'
        self.num_experts = len(expert_model_dump)
        self.experts = nn.ModuleList([MoeExpert(input_size, model_dump,
                                                use_hard_prediction, trainable=trainable_expert)
                                      for model_dump in expert_model_dump])
        self.pre_gate = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
        )

        self.domain_detector = nn.Linear(64, domain_size)
        self.gate = nn.Linear(64, self.num_experts)

    def forward(self, x):
        pre_gate_logits = self.pre_gate(x)
        domain_outputs = self.domain_detector(pre_gate_logits)
        domain_outputs = torch.softmax(domain_outputs, dim=1)

        gate_logits = self.gate(pre_gate_logits)
        gate_probs = torch.softmax(gate_logits, dim=1)
        #gate_probs = F.gumbel_softmax(gate_scores, tau=self.temperature, hard=False, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_outputs = torch.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)

        return MoEResultWithDomain(weighted_outputs, expert_outputs, gate_probs, domain_outputs)


class MaxProbabilityModel(BaseModel):
    def __init__(self, input_size, expert_model_dump):
        super().__init__()
        assert isinstance(input_size, int), 'input_size must be an integer'
        assert isinstance(expert_model_dump, (list, tuple)), 'expert_model_dump must be an list'
        self.num_experts = len(expert_model_dump)
        self.experts = nn.ModuleList([MoeExpert(input_size, model_dump)
                                      for model_dump in expert_model_dump])

    def forward(self, x):
        bs = x.shape[0]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        diffs = torch.abs(expert_outputs - .5).squeeze()
        _, max_indexes = torch.max(diffs, dim=1)
        weighted_outputs = torch.zeros((bs, 1), device=x.device)

        for i in range(bs):
            weighted_outputs[i, 0] = expert_outputs[i, max_indexes[i]]

        return MoEResult(weighted_outputs,
                         expert_outputs,
                         torch.tensor([0], device=x.device))


class StackingCombiner(BaseModel):
    def __init__(self, input_size, expert_model_dump):
        super().__init__()
        assert isinstance(input_size, int), 'input_size must be an integer'
        assert isinstance(expert_model_dump, (list, tuple)), 'expert_model_dump must be an list'
        self.num_experts = len(expert_model_dump)
        self.experts = nn.ModuleList([MoeExpert(input_size, model_dump)
                                      for model_dump in expert_model_dump])
        self.stacker = nn.Sequential(
            nn.Linear(input_size + self.num_experts, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        x = torch.cat((x, expert_outputs.squeeze()), dim=1)

        y = self.stacker(x)

        return MoEResult(y, expert_outputs, torch.tensor([0], device=x.device))


class MoESparseModel(SparseMoE):
    def __init__(self, input_size, expert_model_dump, use_hard_prediction=False, k=1):
        assert isinstance(input_size, int), 'input_size must be an integer'
        assert isinstance(expert_model_dump, (list, tuple)), 'expert_model_dump must be an list'
        super().__init__(input_size=input_size,
                         output_size=1,
                         num_experts=len(expert_model_dump),
                         hidden_size=128,
                         noisy_gating=True,
                         k=k)
        self.num_experts = len(expert_model_dump)
        self.experts = nn.ModuleList([MoeExpert(input_size, model_dump, use_hard_prediction)
                                      for model_dump in expert_model_dump])

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        weighted_outputs = dispatcher.combine(expert_outputs)
        return MoEResultWithLoss(weighted_outputs, expert_outputs, gates, loss)

