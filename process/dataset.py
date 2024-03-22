import pandas as pd
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset

AA_DICTS = {'<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,
            'B': 25,
            'U': 26,
            'Z': 27,
            'O': 28,
            '.': 29,
            '-': 30,
            '<null_1>': 31,
            '<mask>': 32}


def collate_fn(batch):
    padded_seqs, mask_seqs, padded_secs, mask_secs = zip(*batch)

    # Stack all sequences and masks
    padded_seqs = torch.stack(padded_seqs)
    mask_seqs = torch.stack(mask_seqs)
    padded_secs = torch.stack(padded_secs)
    mask_secs = torch.stack(mask_secs)

    return padded_seqs, mask_seqs, padded_secs, mask_secs


class SequenceDataset(Dataset):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    SEC_STRUCTURE = '-HBEGITS'

    def __init__(self, csv_path, min_len, max_len, return_origin_data: bool = False):
        super().__init__()
        self.max_len = max_len
        self.return_origin_data = return_origin_data
        self.data = self.load_and_preprocess_data(csv_path, min_len, max_len)

    def load_and_preprocess_data(self, csv_path, min_len, max_len):
        df = pd.read_csv(csv_path)
        df = df[df.seq_len.between(min_len, max_len - 2)]
        df['encoded_seq'] = df['seq'].apply(self.encode_sequence, categories=self.AMINO_ACIDS)
        df['encoded_sec'] = df['sec'].apply(self.encode_sequence, categories=self.SEC_STRUCTURE)
        return df

    @staticmethod
    def encode_sequence(sequence, categories):
        cat_to_index = {cat: i for i, cat in enumerate(categories)}
        # Add error handling for unexpected characters
        int_encoded = [cat_to_index.get(cat, 0) for cat in sequence if cat in cat_to_index]
        return torch.tensor(int_encoded, dtype=torch.long)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        int_seq = row.encoded_seq
        int_sec = row.encoded_sec

        # Padding
        padded_seq = pad(int_seq, (0, self.max_len - len(int_seq)), value=0)
        padded_sec = pad(int_sec, (0, self.max_len - len(int_sec)), value=0)

        # Masks
        mask_seq = torch.zeros(self.max_len, dtype=torch.bool)
        mask_sec = torch.zeros(self.max_len, dtype=torch.bool)
        mask_seq[:len(int_seq)] = 1
        mask_sec[:len(int_sec)] = 1

        if self.return_origin_data:
            pdb = row.name
            seq = row.seq
            sec = row.sec
            return padded_seq, mask_seq, padded_sec, mask_sec, seq, sec, pdb
        return padded_seq, mask_seq, padded_sec, mask_sec

    def __len__(self):
        return len(self.data)


class SolubilityDataset(Dataset):
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
    def __init__(self, csv_path, min_len, max_len, return_origin_data: bool = False):
        super().__init__()
        self.max_len = max_len
        self.return_origin_data = return_origin_data
        self.data = self.load_and_preprocess_data(csv_path, min_len, max_len)

    def load_and_preprocess_data(self, csv_path, min_len, max_len):
        df = pd.read_csv(csv_path)
        df = df[df.seq_len.between(min_len, max_len - 2)]
        df['encoded_seq'] = df['seq'].apply(self.encode_sequence, categories=self.AMINO_ACIDS)
        return df

    @staticmethod
    def encode_sequence(sequence, categories):
        cat_to_index = {cat: i for i, cat in enumerate(categories)}
        # Add error handling for unexpected characters
        int_encoded = [cat_to_index.get(cat, 0) for cat in sequence if cat in cat_to_index]
        return torch.tensor(int_encoded, dtype=torch.long)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        int_seq = row.encoded_seq
        label = row.y

        # Padding
        padded_seq = pad(int_seq, (0, self.max_len - len(int_seq)), value=20)

        # Masks
        mask_seq = torch.zeros(self.max_len, dtype=torch.bool)
        mask_seq[:len(int_seq)] = 1

        if self.return_origin_data:
            pdb = row.name
            seq = row.seq
            return padded_seq, mask_seq, label,seq, pdb
        return padded_seq, mask_seq, label

    def __len__(self):
        return len(self.data)