from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm

from .layer import *


def get_jumps(timesteps, jumps_every:int=100, r:int=5) -> List[int]:
    jumps = []
    for i in range(0, torch.max(timesteps), jumps_every):
        jumps.extend([i] * r)
    jumps.reverse()  # must be in descending order
    return jumps


def log_min_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
    y = a + torch.log1p(-torch.exp(b - a) + epsilon)
    return y


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL divergence between categorical distributions parameterized by logits."""
    out = F.softmax(logits1 + eps, dim=-1) * (
        F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1)
    )
    return out.sum(dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    """KL divergence between categorical distributions parameterized by probs."""
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return out.sum(dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data."""
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x.to(torch.long), logits.shape[-1])
    return (log_probs * x_onehot).sum(dim=-1)


def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(dim=tuple(range(1, len(x.shape))))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """Create a pre-computed beta schedule for the diffusion process."""
    if schedule == "linear":
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clip(betas, min=0, max=0.999)
    elif schedule == "squaredcos_cap_v2":
        return betas_for_alpha_bar(n_timestep, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

    return betas


def betas_for_alpha_bar(n_timestep, alpha_bar, max_beta=0.999):
    """Create beta schedule that produces a given alpha_bar function."""
    betas = []
    for i in range(n_timestep):
        t1 = i / n_timestep
        t2 = (i + 1) / n_timestep
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


class TransitionMatrix(nn.Module):
    def __init__(self, num_classes, schedule, transition_mat_type, num_steps):
        super().__init__()
        self.num_classes = num_classes
        self.transition_mat_type = transition_mat_type
        betas = make_beta_schedule(schedule, num_steps)
        self.register_buffer('betas', betas)

    def _get_amino_acid_transition_mat(self, t):
        beta_t = self.betas[t].float()  # 将beta_t转换为float32类型

        amino_acid_groups = [
            [0, 5, 7, 11, 13],    # 非极性
            [1, 2, 9, 15, 16],    # 极性
            [3, 6, 8, 19],        # 酸性
            [4, 12, 14, 17, 18],  # 碱性
            [10]                  # 芳香族
        ]

        # 初始化矩阵，所有元素设为相同的基础值
        base_value = (1 - beta_t) / (self.num_classes - 1)
        mat = torch.full((self.num_classes, self.num_classes), base_value)
        # 设置对角线元素的值
        mat.fill_diagonal_(beta_t)

        for group in amino_acid_groups:
            group_tensor = torch.tensor(group)
            # 获取该组以外的索引
            non_group_indices = [i for i in range(self.num_classes) if i not in group]

            # 增加同一组内的转移概率
            for i in group:
                mat[i, group_tensor] += base_value  # 增加
                mat[i, group_tensor] = torch.clamp(mat[i, group_tensor], max=1.0)  # 限制最大值为1

            # 减少到非组成员的转移概率
            for i in group:
                mat[i, non_group_indices] -= base_value / len(non_group_indices) * len(group)
                mat[i, non_group_indices] = torch.clamp(mat[i, non_group_indices], min=0.0)  # 限制最小值为0

        # 重新归一化每一行，确保每一行的和为1
        row_sums = mat.sum(dim=1, keepdim=True)
        mat /= row_sums
        return mat

    def _get_physicochemical_transition_mat(self, t):
        beta_t = self.betas[t]

        # 氨基酸的一些物化性质,如疏水性、等电点等
        hydrophobicity = torch.tensor([0.62, -0.5, -0.5, -0.9, -3.9, 1.19, -3.5, -0.78, -0.4, 4.2,
                                    0.29, 3.8, -3.5, -0.76, -3.5, -0.18, -0.05, 1.08, 2.8, -1.3])
        isoelectric_point = torch.tensor([6.0, 5.4, 3.2, 2.9, 7.6, 5.7, 5.6, 3.6, 7.5, 6.0,
                                        5.3, 5.9, 5.7, 6.0, 7.6, 5.7, 5.6, 5.9, 5.9, 6.0])
        # 归一化
        hydrophobicity = (hydrophobicity - hydrophobicity.mean()) / hydrophobicity.std()
        isoelectric_point = (isoelectric_point - isoelectric_point.mean()) / isoelectric_point.std()

        # 计算氨基酸之间的欧氏距离
        distances = torch.cdist(torch.stack([hydrophobicity, isoelectric_point]).T,
                                torch.stack([hydrophobicity, isoelectric_point]).T)

        # 将距离转化为相似度(距离越近相似度越高)
        similarities = torch.exp(-distances / distances.mean())

        # 将相似度转化为转移概率矩阵
        mat = beta_t * similarities / similarities.sum(1, keepdim=True)

        diagonal = 1. - mat.sum(1)
        mat += torch.diag(diagonal)
        return mat

    def _get_blosum_transition_mat(self, t):
        beta_t = self.betas[t]

        # 使用BLOSUM62矩阵作为转移矩阵的基础
        blosum62 = torch.tensor([
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]
        ])

        # 将BLOSUM62矩阵转化为概率矩阵
        mat = torch.exp(blosum62) * beta_t
        mat /= mat.sum(1, keepdim=True)

        diagonal = 1. - mat.sum(1)
        mat += torch.diag(diagonal)
        return mat

    def _get_full_transition_mat(self, t):
        beta_t = self.betas[t]
        mat = torch.full((self.num_classes, self.num_classes),
                         beta_t / self.num_classes)
        diagonal = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        mat.fill_diagonal_(diagonal)
        return mat

    def _get_absorbing_transition_mat(self, t):
        beta_t = self.betas[t]
        diagonal = torch.full((self.num_classes,), 1. - beta_t)
        mat = torch.diag(diagonal)
        mat[:, self.num_classes // 2] += beta_t
        return mat

    def _get_gaussian_transition_mat(self, t):
        transition_bands = self.num_classes - 1
        beta_t = self.betas[t]

        values = torch.linspace(0., self.num_classes, self.num_classes)
        values = values * 2. / (self.num_classes - 1.)
        values = -values[:transition_bands + 1] ** 2 / beta_t
        values = torch.cat([values.flip(0), values[1:]], dim=0)
        values = torch.softmax(values, dim=0)[transition_bands:]

        mat = torch.zeros((self.num_classes, self.num_classes))
        for k in range(1, transition_bands + 1):
            off_diag = values[k]
            mat += torch.diag(off_diag.repeat(self.num_classes - k), k)
            mat += torch.diag(off_diag.repeat(self.num_classes - k), -k)

        diagonal = 1. - mat.sum(1)  # Ensure rows sum to one
        mat += torch.diag(diagonal)
        return mat

    def forward(self, t):
        if self.transition_mat_type == 'uniform':
            return self._get_full_transition_mat(t)
        elif self.transition_mat_type == 'gaussian':
            return self._get_gaussian_transition_mat(t)
        elif self.transition_mat_type == 'absorbing':
            return self._get_absorbing_transition_mat(t)
        elif self.transition_mat_type == 'amino_acid_group':
            return self._get_amino_acid_transition_mat(t)
        elif self.transition_mat_type == 'physicochemical':
            return self._get_physicochemical_transition_mat(t)
        elif self.transition_mat_type == 'blosum':
            return self._get_blosum_transition_mat(t)
        else:
            raise ValueError(f"transition_mat_type must be one of 'uniform', 'gaussian', 'absorbing', "
                             f"'amino_acid_group', 'physicochemical', 'blosum', but is {self.transition_mat_type}")


class DiscreteDiffusion(nn.Module):
    def __init__(self, num_steps:int, num_classes:int, schedule:str, transition_type:str,
                 d_model:int, num_heads:int, num_layers:int, max_seq_length:int,
                 loss_type:str, hybrid_coeff:float=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.transition_type = transition_type
        self.loss_type = loss_type
        self.hybrid_coeff = hybrid_coeff
        self.eps = 1e-6

        self.transition_mat = TransitionMatrix(num_classes, schedule, transition_type, num_steps)
        self.denoise_model = ProteinDenoiser(d_model, num_heads, num_layers, max_seq_length,
                                             num_classes=num_classes, num_ss_classes=9, dropout=0.1)

        q_onestep_mats = self._construct_q_mats(num_steps)
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, num_steps):
            q_mat_t = q_mat_t @ q_onestep_mats[t]  # Q_{1...t} = Q_{1...t-1} Q_t
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        q_onestep_mats_t = q_onestep_mats.transpose(1, 2)
        self.register_buffer('q_onestep_mats', q_onestep_mats)
        self.register_buffer('q_mats', q_mats)
        self.register_buffer('q_onestep_mats_t', q_onestep_mats_t)
        self.register_buffer('_dummy', torch.empty([0,]))

    def _construct_q_mats(self, num_steps):
        q_onestep_mats = torch.stack([self.transition_mat(t) for t in range(num_steps)])
        return q_onestep_mats

    def q_sample(self, x_0, t, noise):
        logits = torch.log(self.q_probs(x_0, t, self.q_mats) + self.eps)
        gumbel_noise = -torch.log(-torch.log(torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)))
        return torch.argmax(logits + gumbel_noise.view(logits.shape), dim=-1)

    def q_probs(self, x_0, t, trans_mat):
        x_0 = F.one_hot(x_0, self.num_classes).float() if len(x_0.shape) == 2 else x_0
        q_mats_t = trans_mat[t]
        return x_0 @ q_mats_t  # (B, L, N)

    def q_posterior_logits(self, x_0, x_t, t, x_start_logits=False):
        fact1 = self.q_probs(x_t, t, self.q_onestep_mats_t)
        t_1 = torch.where(t == 0, t, t - 1)
        if x_start_logits:
            fact2 = self.q_probs(F.softmax(x_0, dim=-1), t_1, self.q_mats)
            zero_logits = x_0
        else:
            fact2 = self.q_probs(x_0, t_1, self.q_mats)
            zero_logits = torch.log(F.one_hot(x_0.long(), self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.view([t.shape[0]] + [1] * (len(out.shape)-1))  # (B, 1, 1)
        return torch.where(t_broadcast == 0, zero_logits, out)

    def get_model_probs(self, x_t, t, padding_mask=None, **denoise_kwargs):
        pred_x_start_logits = self.denoise_model(x_t, t, padding_mask=padding_mask, **denoise_kwargs)
        t_broadcast = t.view([t.shape[0]] + [1] * (len(pred_x_start_logits.shape)-1))
        model_logits = torch.where(t_broadcast == 0,
                                   pred_x_start_logits,
                                   self.q_posterior_logits(pred_x_start_logits, x_t, t, x_start_logits=True))
        return model_logits, pred_x_start_logits

    def vb_terms_bpd(self, x_0, x_t, t, padding_mask, **denoise_kwargs):
        true_logits = self.q_posterior_logits(x_0, x_t, t)
        model_logits, pred_x_start_logits = self.get_model_probs(x_t, t, padding_mask, **denoise_kwargs)
        mask = padding_mask.unsqueeze(-1).expand_as(true_logits)
        true_logits = true_logits * mask
        model_logits = model_logits * mask
        kl = categorical_kl_logits(logits1=true_logits, logits2=model_logits)
        kl = meanflat(kl) / np.log(2.)
        decoder_nll = -categorical_log_likelihood(x_0*padding_mask, model_logits)
        decoder_nll = meanflat(decoder_nll) / np.log(2.)

        # At the first timestep return decoder NLL, otherwise return KL(q||p)
        assert kl.shape == decoder_nll.shape == t.shape == (x_0.shape[0],)
        return torch.where(t == 0, decoder_nll, kl), pred_x_start_logits

    def cross_entropy_x_start(self, x_0, pred_x_start_logits, padding_mask):
        mask = padding_mask.unsqueeze(-1).expand_as(pred_x_start_logits)
        ce = -categorical_log_likelihood(x_0*padding_mask, pred_x_start_logits*mask)
        ce = meanflat(ce) / np.log(2.)
        assert ce.shape == (x_0.shape[0],)
        return ce

    def forward(self, x_0, t=None, padding_mask=None, **denoise_kwargs):
        N, L = x_0.shape[:2]
        t = torch.randint(0, self.num_steps, (N,), device=self._dummy.device) if t is None else t
        noise = torch.rand(x_0.shape + (self.num_classes,), device=x_0.device)
        x_t = self.q_sample(x_0, t, noise)

        if self.loss_type == 'kl':
            losses, _ = self.vb_terms_bpd(x_0=x_0, x_t=x_t, t=t, padding_mask=padding_mask,**denoise_kwargs)
        elif self.loss_type == 'cross_entropy_x_start':
            _, pred_x_start_logits = self.get_model_probs(x_t, t, padding_mask, **denoise_kwargs)
            losses = self.cross_entropy_x_start(x_0, pred_x_start_logits, padding_mask=padding_mask)
        elif self.loss_type == 'hybrid':
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(x_0, x_t, t, padding_mask=padding_mask, **denoise_kwargs)
            ce_losses = self.cross_entropy_x_start(x_0, pred_x_start_logits, padding_mask=padding_mask)
            losses = vb_losses + self.hybrid_coeff * ce_losses
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        assert losses.shape == t.shape
        losses = losses.sum()
        return losses

    @torch.no_grad()
    def sample(self, condition, condition_mask, classifier=None, scale=None, original_data=None, keep_mask=None, j=10, r=10):
        """
        A unified sampling function for both standard and inpainting tasks.
        Args:
            condition: Input condition.
            condition_mask: Mask for the condition.
            original_data: Original data for inpainting. Default is None.
            keep_mask: Mask for keeping original data in inpainting. Default is None.
            j: Parameter for noise addition in inpainting. Default is 10.
            r: Parameter for determining jumps in inpainting. Default is 5.
        Returns:
            The sampled output.
        """
        N, L = condition.shape[:2]
        time_steps = torch.from_numpy(np.arange(0, self.num_steps)[::-1].copy())
        jumps = get_jumps(time_steps, r=r) if original_data is not None else []

        s_rand = torch.randint_like(condition, low=0, high=self.num_classes)
        s_t = torch.where(condition_mask, s_rand, 0)
        # for t in tqdm(range(self.num_steps, 0, -1)):
        for t in tqdm(reversed(range(self.num_steps)), total=self.num_steps, desc='Sample'):
            noise = torch.rand((N, L, self.num_classes), device=self._dummy.device)
            noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
            gumbel_noise = -torch.log(-torch.log(noise))
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)
            if original_data is not None:
                while jumps and jumps[0] == t:
                    jumps.pop(0)
                    noise = torch.rand((N, L, self.num_classes), device=self._dummy.device)
                    noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
                    s_t = self.q_sample(s_t, t_tensor, noise)
                    for override_t in range(t + j, t, -1):
                        override_tensor = torch.full([N, ], fill_value=override_t, dtype=torch.long, device=self._dummy.device)
                        noise = torch.rand((N, L, self.num_classes), device=self._dummy.device)
                        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
                        gumbel_noise = -torch.log(-torch.log(noise))
                        model_logits, _ = self.get_model_probs(s_t, override_tensor, ss=condition, padding_mask=condition_mask)
                        s_t = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

                s_known = self.q_sample(original_data, t_tensor, noise)

            model_logits, _ = self.get_model_probs(s_t, t_tensor, ss=condition, padding_mask=condition_mask)
            nonzero_mask = (t_tensor != 0).reshape(s_t.shape[0], *([1] * (len(s_t.shape))))
            # For numerical precision clip the noise to a minimum value
            s_t = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

            if original_data is not None:
                s_unknown = s_t
                s_t = keep_mask * s_known + (1 - keep_mask.long()) * s_unknown

        return s_t.cpu()

def move_to_device(input_list, device):
    return [tensor.to(device) for tensor in input_list]


if __name__ == "__main__":

    device = 'cpu'
    x = torch.randint(0, 20, (16, 256))
    condition = torch.randint(0, 8, (16, 256))
    paddint_mask = torch.randint(0, 2, (16, 256)).float()

    x, condition, paddint_mask = move_to_device([x, condition, paddint_mask], device)

    model = DiscreteDiffusion(
        num_steps=1000,
        num_classes=20,
        schedule='cosine',
        transition_type='blosum',
        d_model=512,
        num_heads=8,
        num_layers=12,
        max_seq_length=256,
        loss_type='hybrid'
    ).to(device)

    loss = model(x, ss=condition, padding_mask=paddint_mask)
    print(loss)