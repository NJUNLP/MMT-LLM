'''Basic Retriever'''

from datasets import Dataset, DatasetDict
from typing import List, Union, Optional, Tuple, Dict
from openicl import IclDatasetReader, IclPromptTemplate
from openicl.utils.check_type import _check_str
from accelerate import Accelerator

class IclBaseRetriever:
    """Basic In-context Learning Retriever Class
        Base class of In-context Learning Retriever, no retrieve method
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
    """    
    select_ds = None
    generation_ds = None
    generation_ds_references = None
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
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        self.dataset_reader = IclDatasetReader._check_dataset_reader(dataset_reader)
        self.ice_template = ice_template
        self.prompt_template = prompt_template
        self.ice_separator = ice_separator
        self.ice_eos_token = ice_eos_token
        self.prompt_eos_token = prompt_eos_token
        self.ice_num = ice_num
        self.select_split = select_split
        self.generation_split = generation_split
        self.accelerator = accelerator
        if isinstance(self.dataset_reader.dataset, Dataset):
            self.select_ds = self.dataset_reader.dataset
            self.generation_ds = self.dataset_reader.dataset
            self.generation_ds_references = self.dataset_reader.dataset[self.dataset_reader.pred_label]
            if self.accelerator is not None:
                self.generation_ds = self.generation_ds.shard(
                                        num_shards=self.accelerator.num_processes,
                                        index=self.accelerator.process_index
                )
        else: 
            self.select_ds = self.dataset_reader.dataset[self.select_split]
            self.generation_ds = self.dataset_reader.dataset[self.generation_split]
            self.generation_ds_references = self.dataset_reader.dataset[self.generation_split][self.dataset_reader.pred_label]
            if self.accelerator is not None:
                self.generation_ds = self.generation_ds.shard(
                                        num_shards=self.accelerator.num_processes,
                                        index=self.accelerator.process_index
                )
            
        
    def retrieve(self) -> List[List]:
        """
            Retrieve for each data in generation_ds.
        Returns:
            List[List]: the index list of in-context example for each data in generation_ds.
        """
        raise NotImplementedError("Method hasn't been implemented yet")
    
    
    def get_labels(self):
        labels = []
        if self.prompt_template is not None:
            labels = list(self.prompt_template.template.keys())[:]
        elif self.ice_template is not None and self.ice_template.ice_token is not None:
            labels = list(self.ice_template.template.keys())[:]
        else:
            labels = list(set(self.generation_ds[self.dataset_reader.pred_label]))
        return labels
        
    
    def generate_ice(self, idx_list: List[int]) -> str:
        generated_ice_list = []
        dr = self.dataset_reader
        for idx in idx_list:
            if self.ice_template is None:
                generated_ice_list.append(' '.join(list(map(str,[self.select_ds[idx][ctx] for ctx in dr.ctx_list] + [self.select_ds[idx][dr.pred_label]]))))
            else:
                generated_ice_list.append(self.ice_template.generate_ice_item(self.select_ds[idx], self.select_ds[idx][dr.pred_label]))
        generated_ice = self.ice_separator.join(generated_ice_list) + self.ice_eos_token
        return generated_ice
    
    def generate_ice_for_multi_retriever(self, idx_list: List[int], rtr_list) -> str:
        generated_ice_list = []
        for idx, rtr in zip(idx_list, rtr_list):
            dr = rtr.dataset_reader
            if rtr.ice_template is None:
                generated_ice_list.append(' '.join(list(map(str,[rtr.select_ds[idx][ctx] for ctx in dr.ctx_list] + [rtr.select_ds[idx][dr.pred_label]]))))
            else:
                generated_ice_list.append(rtr.ice_template.generate_ice_item(rtr.select_ds[idx], rtr.select_ds[idx][dr.pred_label]))
        generated_ice = self.ice_separator.join(generated_ice_list) + self.ice_eos_token
        return generated_ice
    
    def generate_prompt(self, idx: int, ice: str) -> Tuple[List[str], List]:
        prompt_list = []
        labels = []
        if self.prompt_template is not None and isinstance(self.prompt_template.template, Dict):
            labels = list(self.prompt_template.template.keys())[:]
        elif self.ice_template is not None and isinstance(self.ice_template.template, Dict) and self.ice_template.ice_token is not None:
            labels = list(self.ice_template.template.keys())[:]
        else:
            labels = list(set(self.generation_ds[self.dataset_reader.pred_label]))
        for label in labels:
            prompt_list.append(self.generate_label_prompt(idx, ice, label))
        return prompt_list, labels
    
    
    def generate_label_prompt(self, idx: int, ice: str, label) -> str:
        if self.prompt_template is not None:
            return self.prompt_template.generate_label_prompt_item(self.generation_ds[idx], ice, label) + self.prompt_eos_token
        elif self.ice_template is not None and self.ice_template.ice_token is not None:
            return self.ice_template.generate_label_prompt_item(self.generation_ds[idx], ice, label) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(list(map(str,[self.generation_ds[idx][ctx] for ctx in self.dataset_reader.ctx_list])))
            return ice + prefix_prompt + ' ' + str(label) + self.prompt_eos_token
    
    
    def generate_prompt_for_generate_task(self, idx, ice, gen_field_replace_token=''):
        if self.prompt_template is not None:
            return self.prompt_template.generate_item(self.generation_ds[idx], gen_field=self.dataset_reader.pred_label, gen_field_replace_token=gen_field_replace_token, ice_field_replace_token=ice) + self.prompt_eos_token
        elif self.ice_template is not None and self.ice_template.ice_token is not None:
            return self.ice_template.generate_item(self.generation_ds[idx], gen_field=self.dataset_reader.pred_label, gen_field_replace_token=gen_field_replace_token, ice_field_replace_token=ice) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(list(map(str,[self.generation_ds[idx][ctx] for ctx in self.dataset_reader.ctx_list])))
            return ice + prefix_prompt + self.prompt_eos_token

    def shuffle(self, ice_idx_list, permutation):
        for idx, ice_idx in enumerate(ice_idx_list):
            assert len(ice_idx) == len(permutation), "{} != {}".format(ice_idx, permutation)

            new_list = []
            for tgt_idx in permutation:
                new_list.append(ice_idx[tgt_idx])
            ice_idx_list[idx] = new_list