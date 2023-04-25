'''Topk-MDL retriever'''

from openicl import IclDatasetReader, IclPromptTemplate
from openicl.icl_retriever import IclTopkRetriever
from openicl.utils.calculate import entropy
from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM
import tqdm
import torch
import numpy as np
from accelerate import Accelerator

class IclTopkMDLRetriever(IclTopkRetriever):
    """Topk-MDL In-context Learning Retriever Class
        Class of Topk-MDL retriever
    Attributes:
        dataset_reader: an IclDatasetReader instance.
        ice_template: an IclPromptTemplate instance, template used for in-context examples generation. if ice_template.ice_token is not None, used for prompt generation as well.
        prompt_template: an IclPromptTemplate instance, template used for prompt generation.
        ice_separator: str that separates each in-context example.
        ice_eos_token: str that adds to the end of in-context examples.
        prompt_eos_token: str that adds to the end of prompt.
        ice_num: number of data in in-context examples.
        candidate_num: number of data selected in the topk stage.
        select_split: str of name for selection dataset. We select data for in-context examples used data in selection dataset(default is 'train')
        generation_split: str of name for generation dataset. We generate prompt for each data in generation dataset(default is 'validation')
        select_ds: selection dataset. We select data for in-context examples used data in selection dataset.
        generation_ds: generation dataset. We generate prompt for each data in generation dataset.
        device: str of device name.
        batch_size: int, batch size for DataLoader. 
        model: a SentenceTransformer instance, used to calculate embeddings.
        tokenizer: a tokenizer initialized by AutoTokenizer.from_pretrained() method.
        index: index generated with FAISS.
        ce_model_name: str, name of metric_model.
        metric_model: a model initialized by AutoModelForCausalLM.from_pretrained() method, used to calculate MDL.
        topk_random_select_time: 
    """
    metric_model = None
    def __init__(self, 
                 dataset_reader: IclDatasetReader,
                 ice_template: Optional[IclPromptTemplate] = None,
                 prompt_template: Optional[IclPromptTemplate] = None,
                 ice_separator: Optional[str] ='\n',
                 ice_eos_token: Optional[str] ='\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name : Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 select_split: Optional[str] = 'train',
                 generation_split: Optional[str] = 'validation',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 topk_random_select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, ice_template, prompt_template, ice_separator, ice_eos_token, prompt_eos_token, sentence_transformers_model_name, ice_num, select_split, generation_split, tokenizer_name, batch_size, accelerator)
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.topk_random_select_time = topk_random_select_time

        
    def topk_search(self):
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]
        for entry in tqdm.tqdm(res_list):
            idx = entry['metadata']['id']
            # data = self.generation_ds[idx]
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.select_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []
            for _ in range(self.topk_random_select_time):
                rand_idx_list = np.random.choice(near_ids, self.ice_num, replace=False)
                candidates.append(rand_idx_list)
                input_texts = []
                for rand_idx in rand_idx_list:
                    input_texts.append(self.select_datalist[rand_idx])
                loss_list = self.cal_ce(input_texts)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                neg_entropy = -entropy(normalized_probs, label_dim=0)
                mdl_scores.append(neg_entropy)
            
            # print(mdl_scores)
            rtr_idx_list[idx] = candidates[mdl_scores.index(max(mdl_scores))]
            rtr_idx_list[idx] = [int(i) for i in rtr_idx_list[idx]]
            
        return rtr_idx_list   
        
        
    def retrieve(self):
        return self.topk_search()
        
        
    def cal_ce(self, input_texts: List[List]):
        if self.metric_model is None:
            print("load metric model")
            self.metric_model = AutoModelForCausalLM.from_pretrained(self.ce_model_name)
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.metric_model(**inputs)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())
        
        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
    