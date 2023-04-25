'''chain-of-thought inferencer'''

import torch
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer import IclBaseInferencer
from typing import List, Union, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.icl_dataset_reader import DatasetEncoder
from accelerate import Accelerator

class IclCOTInferencer(IclBaseInferencer):
    """COT In-context Learning Inferencer Class
        Chain-of-Thought class of In-context Learning Inferencer
    Attributes:
        retriever: an IclBaseRetriever instance.
        cot_list: list of str, chain-of-thought specific sentence.
        metric: an IclBaseEvaluator instance.
        model: inferencer LM, could be initialized by name only or a config class. 
        tokenizer: a tokenizer initialized by AutoTokenizer.from_pretrained() method.
        max_model_token_num: LM maximum tokenized words allowed. 
        batch_size: int, batch size for DataLoader. 
        gen_field_replace_token: str that used to replace generation field token when generate prompts.
        generation_kwargs: dict, generate() method parameters. 
    """
    def __init__(self, 
                 retriever: IclBaseRetriever, 
                 cot_list: Optional[List[str]] = [],
                 metric: Optional[Union[str, IclBaseEvaluator]] = 'acc', 
                 references: Optional[List] = None, 
                 model_name: Optional[str] = 'gpt2-xl', 
                 tokenizer_name: Optional[str] = None, 
                 max_model_token_num: Optional[int] = None, 
                 model_config: Optional[PretrainedConfig] = None, 
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs = {"max_new_tokens": 100,
                                      "do_sample": False},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 api_name: Optional[str] = None
    ) -> None:
        super().__init__(retriever, metric, references, model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, api_name)
        self.cot_list = cot_list
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs

    
    def inference(self, src_lang, tgt_lang):
        predictions = [0 for i in range(len(self.retriever.generation_ds))]
        prompt_list = []
        ice_idx_list = self.retriever.retrieve()
        cot_first_str = '' if len(self.cot_list) == 0 else str(self.cot_list[0])
        for idx, ice_idx in enumerate(tqdm(ice_idx_list)):
            ice = self.retriever.generate_ice(ice_idx)
            prompt = self.retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=self.gen_field_replace_token + cot_first_str)
            if self.max_model_token_num is not None:
                prompt_token_num = self.get_input_token_num(prompt)
                while len(ice_idx) > 0 and prompt_token_num > self.max_model_token_num:
                    ice_idx = ice_idx[:-1]
                    ice = self.retriever.generate_ice(ice_idx)
                    prompt = self.retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=self.gen_field_replace_token + cot_first_str)
                    prompt_token_num = self.get_input_token_num(prompt)
            prompt_list.append(prompt)
            if idx == 0:
                print(f'prompt:\n {prompt}')
        for cot_idx in range(1, len(self.cot_list)):
            encode_data = DatasetEncoder(prompt_list, tokenizer=self.tokenizer)
            co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
            dataloader = DataLoader(encode_data, batch_size=self.batch_size, collate_fn=co)
            
            for idx, entry in enumerate(tqdm(dataloader)):
                metadata = entry.pop("metadata")
                print(entry)
                outputs = self.model.generate(input_ids=entry.input_ids,
                                            attention_mask=entry.attention_mask,
                                            eos_token_id=self.tokenizer.encode("\n")[0],
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            **self.generation_kwargs)
                    
                for mdata, output in zip(metadata, outputs.tolist()):
                    generated = self.tokenizer.decode(output[:], skip_special_tokens=True)
                    generated = generated.strip(self.tokenizer.pad_token)
                    prompt_list[mdata["id"]] = generated + self.cot_list[cot_idx]
                    # print(f"generated:\n{generated}")
        
        encode_data = DatasetEncoder(prompt_list, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_data, batch_size=self.batch_size, collate_fn=co)
        
        for idx, entry in enumerate(tqdm(dataloader)):
            metadata = entry.pop("metadata")
            print(entry)
            outputs = self.model.generate(input_ids=entry.input_ids,
                                        attention_mask=entry.attention_mask,
                                        eos_token_id=self.tokenizer.encode("\n")[0],
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **self.generation_kwargs)
                
            prompt_len = int(entry.attention_mask.shape[1])
            for mdata, output in zip(metadata, outputs.tolist()):
                generated = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                generated = generated.strip(self.tokenizer.pad_token).strip('\n').strip()
                predictions[mdata["id"]] = generated
                print(f"generated:\n{generated}")
        return predictions
