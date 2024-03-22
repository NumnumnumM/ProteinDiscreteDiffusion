import argparse
import importlib

import torch
import torch.nn.functional as F
from models import DiscreteDiffusion
from process import *

SEC_STRUCTURE = '-HBEGITS'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def condition_process(condition:str, max_len:int, num_sample:int, seq_string:str) -> tuple:
    def encode_sequence(sequence, categories):
        cat_to_index = {cat: i for i, cat in enumerate(categories)}
        int_encoded = [cat_to_index.get(cat, 0) for cat in sequence]
        return torch.tensor(int_encoded, dtype=torch.long)
    # condition = condition.replace('L', '-')
    int_condition = encode_sequence(condition, seq_string)

    padded_value = len(seq_string) - 1
    padded_condition = F.pad(int_condition, (0, max_len - len(int_condition)), value=padded_value)

    # Masks
    mask_condition = torch.zeros(max_len, dtype=torch.bool)
    mask_condition[padded_condition != padded_value] = 1

    return padded_condition.expand(num_sample, max_len), mask_condition.expand(num_sample, max_len)

def index_to_sequence(index, seq_len):
    num_sample = index.shape[0]
    sample = []
    for i in range(num_sample):
        local_index = index[i][:seq_len]
        sequence = ''.join([AMINO_ACIDS[j] for j in local_index])
        sample.append(sequence)
    return sample

def sample_sequence(sample_fn:callable, condition, condition_mask, seq_len, classifier,
                    scale, original_data=None, keep_mask=None):
    result = sample_fn(condition,
                       condition_mask,
                       classifier    = classifier,
                       scale         = scale,
                       original_data = original_data,
                       keep_mask     = keep_mask)
    sequence = index_to_sequence(result, seq_len)
    return sequence

def run(condition:str, cache_time:str, original_data=None, num_sample=None, classifier=False, scale:float=10):
    config = get_config(f'./cache/{cache_time}/config')
    seq_len = len(condition)
    if num_sample == None:
        num_sample = config.num_sample
    padded_condition, mask_condition = condition_process(condition, config.dataset.max_len, num_sample, SEC_STRUCTURE)

    if original_data is not None:
        original_data, keep_mask = condition_process(original_data, config.dataset.max_len, num_sample, AMINO_ACIDS)

    device = config.device
    # diffusion_module = importlib.import_module(f'cache.{cache_time}.models.diffusion')
    # MultinomialDiffusion = getattr(diffusion_module, 'MultinomialDiffusion')
    # ------------------------------------
    diffusion = DiscreteDiffusion(
        num_steps        = config.model.num_steps,
        num_classes      = config.model.num_classes,
        schedule         = config.model.schedule,
        transition_type  = config.model.transition_type,
        d_model          = config.model.emb_dim,
        num_heads        = config.model.num_heads,
        num_layers       = config.model.num_layers,
        max_seq_length   = config.model.seq_len,
        loss_type        = config.model.loss_type
    ).to(device)
    state_dict = torch.load(f'./cache/{cache_time}/latest.pt', map_location=device)['model']
    diffusion.load_state_dict(state_dict)
    # ------------------------------------
    padded_condition, mask_condition = move_to_device([padded_condition, mask_condition], device)
    if original_data is not None:
        original_data, keep_mask = move_to_device([original_data, keep_mask], device)
        result = sample_sequence(diffusion.sample, padded_condition, mask_condition, seq_len, classifier, scale, original_data, keep_mask)
    else:
        result = sample_sequence(diffusion.sample, padded_condition, mask_condition, seq_len, classifier, scale)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='--EEEEEEEEEEEEEETTEEEEEEEEEEE--HHHHHHHHHHHHHH--', help='Provide a secondary structure sequence as input. Note: Please replace the secondary structure character "-" with "L".')
    parser.add_argument('--weights', type=str, default='03-15-10', help='Model weight')
    parser.add_argument('--num_sample', type=int, default=16)
    parser.add_argument('--origin_data', type=str, default=None)
    parser.add_argument('--classifier', type=bool, default=True)
    parser.add_argument('--scale', type=float, default=10)

    args = parser.parse_args()
    result = run(args.condition, args.weights, args.origin_data,
                 args.num_sample, args.classifier, args.scale)
    print(result)