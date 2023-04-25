'''Topk Retriever'''

from openicl import IclDatasetReader, IclPromptTemplate
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import IclBaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
import torch
from torch.utils.data import DataLoader
from typing import List, Union, Optional, Tuple
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import faiss
import numpy as np
from accelerate import Accelerator

class IclTopkRetriever(IclBaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of topk retriever
    Attributes:
        dataset_reader: a IclDatasetReader instance.
        ice_template: a IclPromptTemplate instance, template used for in-context examples generation. if ice_template.ice_token is not None, used for prompt generation as well.
        prompt_template: a IclPromptTemplate instance, template used for prompt generation.
        ice_separator: str that separates each in-context example.
        ice_eos_token: str that adds to the end of in-context examples.
        prompt_eos_token: str that adds to the end of prompt.
        ice_num: number of data in in-context examples.
        select_split: str of name for selection dataset. We select data for in-context examples used data in selection dataset(default is 'train')
        generation_split: str of name for generation dataset. We generate prompt for each data in generation dataset(default is 'validation')
        select_ds: selection dataset. We select data for in-context examples used data in selection dataset.
        generation_ds: generation dataset. We generate prompt for each data in generation dataset.
        device: str of device name.
        batch_size: int, batch size for DataLoader. 
        model: a SentenceTransformer instance, used to calculate embeddings.
        tokenizer: a tokenizer initialized by AutoTokenizer.from_pretrained method.
        index: index generated with FAISS.
        orcale: whether to oracle retrieve (target sentences are similar to test case's).
    """
    model = None
    def __init__(self,
                 dataset_reader: IclDatasetReader,
                 ice_template: Optional[IclPromptTemplate] = None,
                 prompt_template: Optional[IclPromptTemplate] = None,
                 ice_separator: Optional[str] ='\n',
                 ice_eos_token: Optional[str] ='\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name : Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 select_split: Optional[str] = 'train',
                 generation_split: Optional[str] = 'validation',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 oracle: Optional[bool] = False
    ) -> None:
        super().__init__(dataset_reader, ice_template, prompt_template, ice_separator, ice_eos_token, prompt_eos_token, ice_num, select_split, generation_split, accelerator)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.oracle = oracle
        if self.oracle:
            gen_datalist = self.dataset_reader.generate_label_field_corpus(self.generation_ds)
        else:
            gen_datalist = self.dataset_reader.generate_ctx_field_corpus(self.generation_ds)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)
        
        
        print("load sentence transformers")
        self.model = SentenceTransformer(sentence_transformers_model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()
        
        
    def create_index(self):
        if self.oracle:
            self.select_datalist = self.dataset_reader.generate_label_field_corpus(self.select_ds)
        else:
            self.select_datalist = self.dataset_reader.generate_ctx_field_corpus(self.select_ds)
        encode_dataset = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_dataset, batch_size=self.batch_size, collate_fn=co)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))   
        res_list = self.forward(dataloader)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        self.embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(self.embed_list, id_list)
        return index

        
    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]
        for entry in tqdm.tqdm(res_list):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
            rtr_idx_list[idx] = near_ids
        return rtr_idx_list
    
    
    def forward(self, dataloader):
        res_list = []
        for _, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list
    
    
    def retrieve(self):
        return self.knn_search(self.ice_num)
