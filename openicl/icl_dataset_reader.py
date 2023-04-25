'''Simple Dataset Reader '''

from typing import List, Union, Optional
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets.splits import NamedSplit
from openicl.icl_prompt_template import IclPromptTemplate
from openicl.utils.check_type import _check_dataset, _check_list, _check_type_list, _check_str
import random
import torch


class IclDatasetReader:
    """In-conext Learning Dataset Reader Class
        Generate an IclDatasetReader instance through 'dataset'.
    Attributes:
        dataset: a datasets.Dataset or datasets.DatasetDict instance.
        ctx_list: a list of column name in the dataset which indicates the context field.
        pred_label: a str of column name in the dataset which indicates the prediction field.
        ds_size: optional; Randomly return ds_size(int) pieces of data when ds_size >= 1, or randomly return int(len(dataset) * ds_size) pieces of data when 0 < ds_size < 1. 
    """
    dataset = None
    ctx_template = None
    label_template = None
    ctx_label_template = None
    def __init__(self, 
                 dataset: Union[Dataset, DatasetDict, str], 
                 ctx_list: List, 
                 pred_label: str, 
                 name: Optional[str] = None,
                 data_files: Optional[str] = None,
                 ctx_template: Optional[IclPromptTemplate] = None,
                 label_template: Optional[IclPromptTemplate] = None,
                 ctx_label_template: Optional[IclPromptTemplate] = None,
                 ds_size: Union[None, int, float] = None,
                 split: Optional[NamedSplit] = None
    ) -> None:
        self.ctx_list = _check_list(ctx_list)
        self.pred_label = _check_str(pred_label)
        self.ds_size = _check_type_list(ds_size, [None, int, float])
        if ctx_template is not None:
            self.ctx_template = IclPromptTemplate._check_prompt_template(ctx_template)
        if label_template is not None:
            self.label_template = IclPromptTemplate._check_prompt_template(label_template)
        if ctx_label_template is not None:
            self.ctx_label_template = IclPromptTemplate._check_prompt_template(ctx_label_template)
        if isinstance(dataset, str):
            self.dataset = load_dataset(dataset, name=name, data_files=data_files)
        else:
            self.dataset = _check_dataset(dataset)
        if split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[split]
        if self.ds_size is not None:
            if isinstance(self.dataset, Dataset):
                self.dataset = load_partial_dataset(dataset, size=self.ds_size)
            if isinstance(self.dataset, DatasetDict):
                for ds_name in self.dataset.keys():
                    self.dataset[ds_name] = load_partial_dataset(self.dataset[ds_name], size=self.ds_size)
                    
    
    def generate_ctx_field_prompt(self, entry):
        prompt = None
        if self.ctx_template is None:
            prompt = ' '.join([str(entry[ctx]) for ctx in self.ctx_list])
        else:
            prompt = self.ctx_template.generate_item(entry)
        return prompt
    
    
    def generate_ctx_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None):
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_ctx_field_prompt(entry))
        return corpus
    
    
    def generate_label_field_prompt(self, entry):
        prompt = None
        if self.label_template is None:
            prompt = str(entry[self.pred_label])
        else:
            prompt = self.label_template.generate_item(entry)
        return prompt
        
    
    def generate_label_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None):
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_label_field_prompt(entry))
        return corpus
    
    
    def generate_ctx_label_field_prompt(self, entry):
        prompt = None
        if self.ctx_label_template is None:
            prompt = ' '.join([entry[ctx] for ctx in self.ctx_list] + [str(entry[self.pred_label])])
        else:
            prompt = self.ctx_label_template.generate_item(entry)
        return prompt      
    

    def generate_ctx_label_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None):
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_ctx_label_field_prompt(entry))
        return corpus  
                    
    
    def _check_dataset_reader(obj) -> "IclDatasetReader":
        if isinstance(obj, IclDatasetReader):
            return obj
        else:
            raise TypeError(f"Expected a IclDatasetReader object, but got {obj}") 
        
            
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
    def __repr__(self):
        return f"IclDatasetReader({{\n    dataset: {self.dataset},\n    ctx_list: {self.ctx_list},\n    a_list: {self.pred_label}\n}})"
    
    
def load_partial_dataset(dataset: Dataset, size: Optional[Union[int, float]] = None) -> Dataset:
    total_size = len(dataset)
    if size >= total_size or size <= 0:
        return dataset
    if size > 0 and size < 1:
        size = int(size * total_size)
    rand = random.Random(x=size)
    index_list = list(range(total_size))
    rand.shuffle(index_list)
    dataset = dataset.select(index_list[:size])
    return dataset


class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name = None, tokenizer = None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)
    
    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt')
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                     "text": data}
            })
            
            
    def __len__(self):
        return self.datalist_length
    
    
    def __getitem__(self, idx):
        return self.encode_dataset[idx]
    

class IndexListReader(torch.utils.data.Dataset):
    def __init__(self, idx_list: List) -> None:
        self.idx_list = idx_list
        self.data_length = len(idx_list)
        self.idx_reader = []
        self.init_dataset()
        
    
    def init_dataset(self):
        for idx in range(self.data_length):
            self.idx_reader.append({
                'metadata' :{
                    'idx_list': self.idx_list[idx],
                    'idx': idx
                }
            })
            
            
    def __len__(self):
        return self.data_length
    
    
    def __getitem__(self, idx):
        return self.idx_reader[idx]
            