import os
import torch
import torch.nn.functional as F
import numpy as np

from transformers import T5Tokenizer, T5EncoderModel
import re

import datasets
import transforms

class ToTensor:
    def __call__(self, x:np.ndarray):
        return torch.from_numpy(x)

class ToOneHotTensor:
    def __init__(self, num_classes:int):
        self.num_classes = num_classes

    def __call__(self, t:np.ndarray):
        t = torch.from_numpy(t)
        t = t.long()
        return F.one_hot(t, self.num_classes)


## extracted data preparation code here
    
class PrepareEmbeddings:

    def __init__(self):

        # Preparing the embedding model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

        # Load the model
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(self.device)

        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        self.model.to(torch.float32) if self.device==torch.device("cpu") else None

    def prepare_embeddings(self, input_file:str, encoder_loaded=None, output_file:str=None, is_test:bool=False, mode:int=0):

        """
        mode: 0: original implementation suggested by the reference https://github.com/agemagician/ProtTrans
                 output has shape (batch size x 7 x 1024) mapped to (batch size x 1024) [final shape]
              1: output has shape (batch size x 8 x 1024) as final shape.
        """

        # prepare your protein sequences as a list
        # seqs = ["PRTEINO", "SEQWENCE"]
        # embeddings = generate_embeddings(seqs, tokenizer=tokenizer, model=model)
        # print(embeddings)

        num_classes = len(datasets.arg_dict)

        transform = lambda x: self._generate_embeddings(x, mode=mode)
        target_transform = transforms.ToOneHotTensor(num_classes=num_classes)
        dataset = datasets.ProteinDataset(input_file=input_file, transform=transform, 
                                target_transform=target_transform, encoder=encoder_loaded, is_test=is_test)
        dataset = dataset.prepare_dataset(batch_size=1, output_file=output_file, mode=mode)

        return dataset
    
    def _generate_embeddings(self, seqs, mode:int=0):
        ## Reference: https://github.com/agemagician/ProtTrans
        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer(seqs, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if mode == 1:
            emb_per_protein = embedding_repr.last_hidden_state[:,:8] # shape (batch size x 8 x 1024)
        else:
            # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]) 
            emb = embedding_repr.last_hidden_state[:,:7] # shape (batch size x 7 x 1024)
            
            # if you want to derive a single representation (per-protein embedding) for the whole protein
            emb_per_protein = emb.mean(dim=1) # shape (1024)

        return emb_per_protein
    
class EmbeddingsLoader:

    def load_embeddings(self, input_file:str, is_test:bool=False, transform=None, target_transform=None, mode:int=0):
        label_file = os.path.join(os.path.dirname(input_file), f"labels_{os.path.basename(input_file)}")

        if mode == 1:
            dataset = datasets.FolderEmbeddingsDataset(input_file=input_file, is_test=is_test, transform=transform, target_transform=target_transform)
        else:
            cache_embeddings = np.loadtxt(input_file, delimiter=",")
            cache_labels = np.loadtxt(label_file, delimiter=",", dtype=str if is_test else int)

            dataset = datasets.EmbeddingsDataset(cache_embeddings, cache_labels, transform=transform, target_transform=target_transform if not is_test else None,
                                                 is_test=is_test, mode=mode)

        return dataset


