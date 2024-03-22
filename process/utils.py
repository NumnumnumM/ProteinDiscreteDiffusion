import inspect
import os
import random
import subprocess
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from Bio import AlignIO, SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from omegaconf import OmegaConf


class HighFrequencySequenceFinder:
    def __init__(self, fasta_path, threshold=90):
        self.align = AlignIO.read(fasta_path, 'fasta')
        self.threshold = threshold
        self.high_freq_sites = self._find_high_frequency_sites()

    def _calc_frequency(self, amino, pos):
        column = str(self.align[:, pos])
        count = column.count(amino)
        frequency = 100 * float(count / len(column))
        return frequency

    def _find_high_frequency_sites(self):
        high_freq_sites = {}
        for pos in range(len(self.align[0])):
            aminos = set(self.align[:, pos])
            for amino in aminos:
                if amino == '-':
                    continue
                freq = self._calc_frequency(amino, pos)
                if freq > self.threshold:
                    high_freq_sites.setdefault(pos, []).append((amino, freq))
        return high_freq_sites

    def _concatenate_continuous_sites(self):
        continuous_sequences = []
        sorted_positions = sorted(self.high_freq_sites.keys())

        if not sorted_positions:
            return continuous_sequences

        current_sequence = [sorted_positions[0]]
        for pos in sorted_positions[1:]:
            if pos == current_sequence[-1] + 1:
                current_sequence.append(pos)
            else:
                continuous_sequences.append(current_sequence)
                current_sequence = [pos]
        continuous_sequences.append(current_sequence)

        return continuous_sequences

    def _get_idx_dict(self):
        tag = 0
        result = {}
        for i, amino in enumerate(list(self.align[0])):
            if amino == '-':
                tag += 1
            else:
                result[i] = i - tag
        seq_lst = list(self.align[0])
        seq_str = ''.join(seq_lst).replace('-', '')
        return result, len(seq_str)

    def get_mask_seq(self):
        idx_dict, origin_length = self._get_idx_dict()
        high_freq_sequences_with_indexes = self.get_high_freq_sequences_with_indexes()
        mask_seq = list('X' * origin_length)
        for seq, start, end in high_freq_sequences_with_indexes:
            try:
                start = idx_dict[start-1]
                end = idx_dict[end-1]
            except:
                print(f'the amion acids:{seq} local:{start}:{end} not in sequence')
                continue
            if start == end:
                mask_seq[start] = seq
            else:
                mask_seq[start:end+1] = seq
        mask_seq = ''.join(mask_seq)
        return mask_seq

    def get_high_freq_sequences_with_indexes(self):
        continuous_sequences = self._concatenate_continuous_sites()
        sequences_with_indexes = []
        for sequence in continuous_sequences:
            seq = ''.join([self.high_freq_sites[pos][0][0] for pos in sequence])
            start_index = sequence[0] + 1  # 转换为1-based索引
            end_index = sequence[-1] + 1  # 转换为1-based索引
            sequences_with_indexes.append((seq, start_index, end_index))
        return sequences_with_indexes


def fold_and_savePDB(seq_lst:Union[list, str], save_path:str, device='cuda:0')->None:
    tag_time = time.strftime("%m-%d-%H-%S", time.localtime())

    save_path = os.path.join(save_path, tag_time)
    os.makedirs(save_path, exist_ok=True)
    fasta_file = os.path.join(save_path, 'sequence.fasta')
    print("Writing FASTA file:", fasta_file)

    with open(fasta_file, mode='w+') as f:
        for idx, sequence in enumerate(seq_lst):
            f.write(f'>{idx}\n')
            f.write(f'{sequence}\n')

    print("Now run OmegaFold....")
    subprocess.run(["omegafold", fasta_file, save_path, f"--device={device}", "--num_cycle=16"])
    print("Done OmegaFold")


def fasta_fold(fasta_file:str, save_path:str, device:str='cuda:0')->None:
    tag_time = time.strftime("%m-%d-%H-%S", time.localtime())
    save_path = os.path.join(save_path, tag_time)
    print("Now run OmegaFold....")
    subprocess.run(["omegafold", fasta_file, save_path, f"--device={device}", "--num_cycle=16"])
    print("Done OmegaFold")


def select_sequence(input_file:str, output_file:str, select_idx:list, file_format:str='fasta'):
    data = SeqIO.parse(input_file, file_format)
    select = [seq for i, seq in enumerate(data) if i in select_idx]
    SeqIO.write(select, output_file, format=file_format)


def calculate_ss_percentages_list(sec_structure):
    ss_types = ['H', 'E', 'T', '~', 'B', 'G', 'I', 'S']
    total_length = len(sec_structure)
    ss_percentages = [(sec_structure.count(ss) / total_length) for ss in ss_types]

    return ss_percentages


def plot_structure_comparison(sec_structure, expected_frequencies, plot_error=False):
    # Define the secondary structure types
    ss_types = ['H', 'E', 'T', '~', 'B', 'G', 'I', 'S']

    # Calculate the frequency of each secondary structure type in the sequence
    actual_frequencies = np.array([sec_structure.count(ss) / len(sec_structure) for ss in ss_types])

    # Calculate the absolute error between expected and actual frequencies
    errors = np.abs(actual_frequencies - expected_frequencies)
    print("Absolute error per SS structure type:", dict(zip(ss_types, errors)))

    # Plot the expected vs actual frequencies for each secondary structure type
    x_positions = np.arange(len(ss_types))
    plt.figure(figsize=(6, 3))
    plt.bar(x_positions - 0.15, expected_frequencies, width=0.3, color='b', align='center', label='Expected')
    plt.bar(x_positions + 0.15, actual_frequencies, width=0.3, color='r', align='center', label='Predict')
    plt.xticks(x_positions, ss_types)
    plt.ylim([0, 1])
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Secondary Structure Prediction Comparison')
    plt.show()

    # Optionally, plot the error
    if plot_error:
        plt.figure(figsize=(6, 3))
        plt.plot(errors, 'o-', label='Error over SS type')
        plt.ylabel('Error')
        plt.xticks(x_positions, ss_types)
        plt.legend()
        plt.title('Error in Secondary Structure Prediction')
        plt.show()

    return errors


def parse_pdb_for_secondary_structure(pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure(pdb_filename, pdb_filename)
    model = structure[0]

    dssp_result = DSSP(model, pdb_filename, file_type='PDB')

    sequence = ''.join(dssp_result[key][1] for key in dssp_result.keys())
    # raw_secondary_structure = ''.join(dssp_result[key][2] for key in dssp_result.keys()).replace('-', '~')
    raw_secondary_structure = ''.join(dssp_result[key][2] for key in dssp_result.keys())

    translation_map = {
        '-': 'C', 'I': 'C', 'T': 'C', 'S': 'C', 'G': 'H', 'B': 'E',
        'H': 'H', 'E': 'E'  # 这些已经是期望的格式
    }

    refined_secondary_structure = ''.join(translation_map[s] for s in raw_secondary_structure)

    return raw_secondary_structure, refined_secondary_structure, sequence


def calculate_sequence_similarity_percentage(seq1, seq2):
    differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
    differences += abs(len(seq1) - len(seq2))

    max_length = max(len(seq1), len(seq2))

    similarity = (max_length - differences) / max_length
    similarity_percentage = similarity * 100

    return similarity_percentage


def move_to_device(input_list, device):
    return [tensor.to(device) for tensor in input_list]


def split_dataset(data, train_percent=0.8, val_percent=0.1, seed=None):
    from torch.utils.data import random_split

    if seed is not None:
        torch.manual_seed(seed)

    total_size = len(data)
    train_size = int(train_percent * total_size)
    val_size = int(val_percent * total_size)
    test_size = total_size - train_size - val_size

    return random_split(data, [train_size, val_size, test_size])


def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def retrieve_name(vars):
    vars_dicts = {}
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for item in vars:
        str_name = [var_name for var_name, var_val in callers_local_vars if var_val is item][0]
        vars_dicts[str_name] = item
    return vars_dicts


def tensorboard_writer(writer, step, data):
    if type(data) == list:
        for key, val in zip(data[0], data[1]):
            dicts = {}
            tag = key.split('_')[-1]
            dicts[key] = data[0][key]
            dicts[val] = data[1][val]
    else:
        for key in data.keys():
            dicts = {}
            tag = key.split('_')[-1]
            dicts[key] = data[key]
    writer.add_scalars(main_tag = tag,
                        tag_scalar_dict = dicts,
                        global_step = step)

    return writer


def convert_list(obj):
    if isinstance(obj, omegaconf.listconfig.ListConfig):
        return list(obj)
    elif isinstance(obj, (tuple, list)):
        return [convert_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_list(value) for key, value in obj.items()}
    else:
        return obj


def get_config(config_path):
    config = OmegaConf.create()
    for file in os.listdir(config_path):
        model_dict_path = os.path.join(config_path, file)
        new_config = OmegaConf.load(model_dict_path)
        config = OmegaConf.merge(config, new_config)
    container = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(convert_list(container))
    return config
