'''votek retriever'''

import os
import json
from openicl import IclDatasetReader, IclPromptTemplate
from openicl.icl_retriever import IclTopkRetriever
from typing import List, Union, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import random
from accelerate import Accelerator

class IclVotekRetriever(IclTopkRetriever):
    """Vote-k In-context Learning Retriever Class
        Class of vote-k retriever
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
        vote_k_k: int, k value of Voke-k Selective Annotation Algorithm
    """
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
                 vote_k_k: Optional[int] = 3,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, ice_template, prompt_template, ice_separator, ice_eos_token, prompt_eos_token, sentence_transformers_model_name, ice_num, select_split, generation_split, tokenizer_name, batch_size, accelerator) 
        self.vote_k_k = vote_k_k
        
        
    def vote_k_select(self, embeddings=None, select_num=None, k=None, overlap_threshold=None, vote_file=None):
        n = len(embeddings)
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file) as f:
                vote_stat = json.load(f)
        else:
            # bar = tqdm(range(n), desc=f'vote {k} selection')
            vote_stat = defaultdict(list)
            # 计算score
            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[idx].append(i)
                # bar.update(1)
            if vote_file is not None:
                with open(vote_file, 'w') as f:
                    json.dump(vote_stat, f)
        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
        j = 0
        selected_indices = []
        while len(selected_indices) < select_num and j < len(votes):
            candidate_set = set(votes[j][1])
            flag = True
            for pre in range(j):
                cur_set = set(votes[pre][1])
                if len(candidate_set.intersection(cur_set)) >= overlap_threshold * len(candidate_set):
                    flag = False
                    break
            if not flag:
                j += 1
                continue
            selected_indices.append(int(votes[j][0]))
            j += 1
        if len(selected_indices) < select_num:
            unselected_indices = []
            cur_num = len(selected_indices)
            for i in range(n):
                if not i in selected_indices:
                    unselected_indices.append(i)
            selected_indices += random.sample(unselected_indices, select_num - cur_num)
        return selected_indices


    def vote_k_search(self):
        self.vote_k_idxs = self.vote_k_select(embeddings=self.embed_list, select_num=self.candidate_num,
                                                  k=self.vote_k_k,overlap_threshold=1)
        print(self.vote_k_idxs)
        return [self.vote_k_idxs[:self.ice_num] for _ in range(len(self.generation_ds))]
    
    
    def retrieve(self):
        return self.vote_k_search()
