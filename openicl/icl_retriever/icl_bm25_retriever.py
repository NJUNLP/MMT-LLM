'''BM25 Retriever'''

from openicl import IclDatasetReader, IclPromptTemplate
from openicl.icl_retriever import IclBaseRetriever
from typing import List, Union, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import trange
from accelerate import Accelerator
from nltk.tokenize import word_tokenize

class IclBM25Retriever(IclBaseRetriever):
    """BM25 In-context Learning Retriever Class
        Class of BM25 retriever
    Attributes:
        dataset_reader: an IclDatasetReader instance.
        ice_template: an IclPromptTemplate instance, template used for in-context examples generation. if ice_template.ice_token is not None, used for prompt generation as well.
        prompt_template: an IclPromptTemplate instance, template used for prompt generation.
        ice_separator: str that separates each in-context example.
        ice_eos_token: str that adds to the end of in-context examples.
        prompt_eos_token: str that adds to the end of prompt.
        ice_num: number of data in in-context examples.
        select_split: str of name for selection dataset. We select data for in-context examples used data in selection dataset(default is 'train')
        generation_split: str of name for generation dataset. We generate prompt for each data in generation dataset(default is 'validation')
        select_ds: selection dataset. We select data for in-context examples used data in selection dataset.
        generation_ds: generation dataset. We generate prompt for each data in generation dataset.
        select_corpus: corpus created by context field data of select_ds.
        generation_corpus: corpus created by context field data of generation_ds.
        bm25: a BM250kapi instance, initialized by select_corpus.
        orcale: whether to oracle retrieve (target sentences are similar to test case's).
    """
    bm25 = None
    select_corpus = None
    def __init__(self,
                 dataset_reader: IclDatasetReader,
                 ice_template: Optional[IclPromptTemplate] = None,
                 prompt_template: Optional[IclPromptTemplate] = None,
                 ice_separator: Optional[str] ='\n',
                 ice_eos_token: Optional[str] ='\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 select_split: Optional[str] = 'train',
                 generation_split: Optional[str] = 'validation',
                 accelerator: Optional[Accelerator] = None,
                 oracle: Optional[bool] = False
    ) -> None:
        super().__init__(dataset_reader, ice_template, prompt_template, ice_separator, ice_eos_token, prompt_eos_token, ice_num, select_split, generation_split, accelerator)
        self.oracle = oracle
        if self.oracle:
            self.select_corpus = [word_tokenize(data) for data in self.dataset_reader.generate_label_field_corpus(self.select_ds)]
        else:
            self.select_corpus = [word_tokenize(data) for data in self.dataset_reader.generate_ctx_field_corpus(self.select_ds)]
        self.bm25 = BM25Okapi(self.select_corpus)
        if self.oracle:
            self.generation_corpus = [word_tokenize(data) for data in self.dataset_reader.generate_label_field_corpus(self.generation_ds)]
        else:
            self.generation_corpus = [word_tokenize(data) for data in self.dataset_reader.generate_ctx_field_corpus(self.generation_ds)]
    
   
    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        for idx in trange(len(self.generation_corpus)):
            query = self.generation_corpus[idx]
            scores = self.bm25.get_scores(query)
            near_ids = list(np.argsort(scores)[::-1][:self.ice_num])
            near_ids = [int(a) for a in near_ids]
            rtr_idx_list.append(near_ids)
        return rtr_idx_list
    