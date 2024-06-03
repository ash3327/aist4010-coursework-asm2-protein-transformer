from Bio import SeqIO
from tqdm import tqdm
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import torch

arg_dict = \
    {'aminoglycoside': 0, 'macrolide-lincosamide-streptogramin': 1, 'polymyxin': 2,
    'fosfomycin': 3, 'trimethoprim': 4, 'bacitracin': 5, 'quinolone': 6, 'multidrug': 7,
    'chloramphenicol': 8, 'tetracycline': 9, 'rifampin': 10, 'beta_lactam': 11,
    'sulfonamide': 12, 'glycopeptide': 13, 'nonarg': 14}
arg_map = dict((v, k) for k, v in arg_dict.items())

class Sequence:
    def __init__(self, input_file:str, is_test:bool=False):
        self._seqs = Sequence.load_dataset(input_file, is_test=is_test)

    @staticmethod
    def load_dataset(input_file:str, is_test:bool=False):
        seqs = list()
        labels = list()
        with open(input_file) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if not is_test:
                    label_t = 'nonarg' if record.id.startswith("sp") else record.id.split("|")[3]
                    label = arg_dict[label_t]
                else:
                    label = record.id
                seqs.append(str(record.seq))
                labels.append(label)

        return seqs, labels
    
    def items(self):
        return self._seqs
    
    def seqs(self):
        return self._seqs[0]
    
    def labels(self):
        return self._seqs[1]
    
    def __len__(self):
        return len(self.labels())

class EncodingScheme:
    def __init__(self, map:list, default='X'):
        self._default = default
        self._map = map
        if self._default not in self._map:
            self._map.append(self._default)
        self._inv_map = {k: v for v, k in enumerate(self._map)}

    def __repr__(self):
        return str({'keys':self._map, 'default':self._default})
    
    def map_to_class(self, seq:str|list):
        return [self._inv_map.get(k, self._default) for k in seq]
    
    def map_from_class(self, seq:list):
        return [self._map[k] for k in seq]
    
    def save(self, output_file:str):
        dir = os.path.dirname(output_file)
        try:
            os.makedirs(dir, exist_ok=True)
        except:
            pass

        series = pd.Series(self._map)
        series['default'] = self._default
        series.to_csv(output_file)

    def get_num_classes(self):
        return len(self._map)

    @classmethod
    def generate_encoding_scheme(cls, seqs:Sequence, output_file:str=None, default='X', log:bool=True):
        seq2 = list()
        if log:
            print("Genearting Encoding Scheme: ")
            print("*Warning: Make sure that you are using consistent encoding schemes for trianing and testing.")
        items = seqs.items()
        items = tqdm(items) if log else items
        for item in items:
            seq, label = item
            for t in seq:
                if t in seq2:
                    continue
                seq2.append(t)
        encoder = cls(seq2, default=default)
        if output_file:
            encoder.save(output_file)
        return encoder
    
    @classmethod
    def load_encoding_scheme(cls, input_file:str):
        series = pd.read_csv(input_file, index_col=0)
        default = series.loc['default'].iloc[0]
        series = series.drop('default')
        map = series.to_dict()
        map = map['0'].values()
        map = list(map)
        return cls(map, default=default)

class ProteinDataset(Dataset):
    def __init__(self, input_file:str, encoder:EncodingScheme=None, log:bool=True, transform=None, target_transform=None, generate:bool=False, is_test:bool=False):
        self.seqs = Sequence(input_file=input_file, is_test=is_test)
        self.encoder = encoder if encoder is not None else EncodingScheme.generate_encoding_scheme(self.seqs, log=log) if generate else None
        self.transform = transform
        self.target_transform = target_transform

        self.prepared = None
        self.batch_size = None
        self.is_test = is_test

    def prepare_dataset(self, batch_size=16, output_file=None, mode:int=0):
        self.batch_size = batch_size

        if mode == 1:
            return self.prepare_dataset_mode_1(output_file=output_file)
        else:
            return self.prepare_dataset_mode_0(output_file=output_file)
        
    def prepare_dataset_mode_0(self, output_file=None):
        self.prepared = list()

        import gc

        print("Preparing Dataset")
        gc.disable()
        for i in tqdm(range(len(self.seqs)//self.batch_size)):
            seq = self.seqs.seqs()[i*self.batch_size:(i+1)*self.batch_size]
            self.prepared.append(self.transform(seq))
        gc.enable()

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            cache_embeddings = torch.concat(self.prepared)
            cache_embeddings = cache_embeddings.cpu().numpy()
            cache_embeddings = np.reshape(cache_embeddings, (len(self),-1))
            print(f"Shape of cache_train: {cache_embeddings.shape}")
            np.savetxt(output_file, cache_embeddings, delimiter=",")

            cache_labels = np.array(self.seqs.labels(), dtype=str if self.is_test else int)
            label_file = os.path.join(os.path.dirname(output_file), f"labels_{os.path.basename(output_file)}")
            np.savetxt(label_file, cache_labels, delimiter=",", fmt="%.u" if not self.is_test else "%s")

        return self
    
    def prepare_dataset_mode_1(self, output_file=None):
        """
        output_file cannot be none, else will be too slow
        """
        self.prepared = list()
        assert output_file is not None

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_dir = output_file.rsplit('.',1)[0]
        os.makedirs(output_dir, exist_ok=True)

        print("Preparing Dataset")
        for i in tqdm(range(len(self.seqs)//self.batch_size)):
            seq = self.seqs.seqs()[i*self.batch_size:(i+1)*self.batch_size]
            cache_embedding = self.transform(seq).cpu().numpy()
            torch.save(cache_embedding, os.path.join(output_dir, f"{i}.dt"))

        cache_labels = np.array(self.seqs.labels(), dtype=str if self.is_test else int)
        label_file = os.path.join(os.path.dirname(output_file), f"labels_{os.path.basename(output_file)}")
        np.savetxt(label_file, cache_labels, delimiter=",", fmt="%.u" if not self.is_test else "%s")    

        dataset = FolderEmbeddingsDataset(input_file=output_file, is_test=self.is_test, transform=None, target_transform=self.target_transform if not self.is_test else None)
        return dataset

    def get_item_from_batch(self, idx):
        return self.prepared[idx//self.batch_size][idx%self.batch_size]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq, label = self.seqs.seqs()[idx], self.seqs.labels()[idx]
        if self.encoder:
            seq = self.encoder.map_to_class(seq)
        if self.prepared:
            seq = self.get_item_from_batch(idx)
        elif self.transform:
            seq = self.transform(seq) # need convert to numpy yourself
        label = self.target_transform(np.array(label)) if not self.is_test else label
        return seq, label


class EmbeddingsDataset(Dataset):

    def __init__(self, embeddings, labels, transform=None, target_transform=None, is_test:bool=False, mode:int=0):
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.is_test = is_test
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq, label = self.embeddings[idx], self.labels[idx]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(np.array(label) if not self.is_test else label)
        return seq, label

class FolderEmbeddingsDataset(Dataset):

    def __init__(self, input_file:str, is_test:bool=False, transform=None, target_transform=None):
        assert input_file is not None

        self.input_dir = input_file.rsplit('.',1)[0]
        self.is_test = is_test
        self.transform = transform
        self.target_transform = target_transform

        label_file = os.path.join(os.path.dirname(input_file), f"labels_{os.path.basename(input_file)}")
        self.labels = np.loadtxt(label_file, delimiter=",", dtype=str if is_test else int)

    def __len__(self):
        return len(self.labels)
    
    def _load_embeddings(self, idx:int):
        return torch.load(os.path.join(self.input_dir, f"{idx}.dt"))

    def __getitem__(self, idx):
        seq, label = self._load_embeddings(idx), self.labels[idx]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(np.array(label) if not self.is_test else label)
        # print(seq, label)
        return seq, label  