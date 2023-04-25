'''Random Retriever'''

from openicl import IclDatasetReader, IclPromptTemplate
from openicl.icl_retriever import IclBaseRetriever
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

class IclRandomRetriever(IclBaseRetriever):
    """Random In-context Learning Retriever Class
        Class of random retriever
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
        seed: int, random seed.
    """
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
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, ice_template, prompt_template, ice_separator, ice_eos_token, prompt_eos_token, ice_num, select_split, generation_split, accelerator)
        self.seed = seed
        
        
    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.select_ds)
        rtr_idx_list = []
        print("retrieve started")
        for _ in trange(len(self.generation_ds)):
            idx_list = np.random.choice(num_idx, self.ice_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        print("retrieve finished")
        return rtr_idx_list
        