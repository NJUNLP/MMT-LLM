'''Generation Inferencer'''

import json
import torch
import os
import numpy as np

from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer import IclBaseInferencer
from openicl.utils.api_service import * 
from typing import List, Union, Optional, Callable
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.icl_dataset_reader import DatasetEncoder
from accelerate import Accelerator

class IclGenInferencer(IclBaseInferencer):
    """Generation In-context Learning Inferencer Class
        Generation class of In-context Learning Inferencer
    Attributes:
        retriever: an IclBaseRetriever instance or a series of IclBaseRetriever.
        metric: an IclBaseEvaluator instance.
        model: inferencer LM, could be initialized by name only or a config class. 
        tokenizer: a tokenizer initialized by AutoTokenizer.from_pretrained() method.
        max_model_token_num: LM maximum tokenized words allowed. 
        batch_size: int, batch size for DataLoader. 
        gen_field_replace_token: str that used to replace generation field token when generate prompts.
        generation_kwargs: dict, generate() method parameters. 
        rtr_order: list, retriever index in retriever_list for each ice.
        ice_operator: callable function operating ice injected into ice composition.
        chatgpt_kwargs: dict, api_get_tokens() parameter if api_name is ChatGPT.
    """
    def __init__(self, 
                 retriever: Union[IclBaseRetriever, List[IclBaseRetriever]], 
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
                 output_json_filename: Optional[str] = None,
                 api_name: Optional[str] = None,
                 rtr_order: Optional[List[int]] = None,
                 ice_operator: Optional[Callable[[List[int], List[IclBaseRetriever]], None]] = None,
                 chatgpt_kwargs = {"api_key_path": None,
                                   "rpm": 20},
                 
    ) -> None:
        if isinstance(retriever, IclBaseRetriever):
            super().__init__(retriever, metric, references, model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, api_name)
            self.retriever_list = [retriever]
        else:
            assert len(retriever) > 0, "Require at least one retriever"
            super().__init__(retriever[0], metric, references, model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, api_name)
            self.retriever_list = retriever
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs
        self.output_json_filename = output_json_filename
        if rtr_order is None:
            self.rtr_order = [0] * self.retriever.ice_num
        else:
            self.rtr_order = rtr_order
        self.ice_operator = ice_operator
        self.chatgpt_kwargs = chatgpt_kwargs
    
    def inference(self, src_lang, tgt_lang):
        tot_num = len(self.retriever.generation_ds)
        if self.accelerator is not None:
            process_data_idx = []
            process_predictions = []
            tot_num *= int(self.accelerator.num_processes)
        predictions = [0 for i in range(tot_num)]
        prompt_list = []
        ice_idx_list = [rtr.retrieve() for rtr in self.retriever_list]
        ice_nums = []
        for idx, ice_idx in enumerate(tqdm(ice_idx_list)):
            ice_idx, rtr = [], []
            for i, rtr_id in enumerate(self.rtr_order):
                ice_idx.append(ice_idx_list[rtr_id][idx][i])
                rtr.append(self.retriever_list[rtr_id])
            if self.ice_operator is not None:
                self.ice_operator(ice_idx, rtr)

            ice = self.retriever.generate_ice_for_multi_retriever(ice_idx, rtr)
            prompt = self.retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=self.gen_field_replace_token)
            if self.max_model_token_num is not None:
                prompt_token_num = self.get_input_token_num(prompt)
                while len(ice_idx) > 0 and prompt_token_num > self.max_model_token_num:
                    ice_idx, rtr = ice_idx[:-1], rtr[:-1]
                    ice = self.retriever.generate_ice_for_multi_retriever(ice_idx, rtr)
                    prompt = self.retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=self.gen_field_replace_token)
                    prompt_token_num = self.get_input_token_num(prompt)
            ice_nums.append(len(ice_idx))
            prompt_list.append(prompt)
        print("Average ice num: ", np.mean(ice_nums))
        # print("Prompt examples: ")
        # for prompt in prompt_list[:3]:
        #     print(prompt)
        
        encode_data = DatasetEncoder(prompt_list, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_data, batch_size=self.batch_size, collate_fn=co)

        # pick up at where it is lost
        if os.path.exists(f"{self.output_json_filepath}/{self.output_json_filename}.json"):
            with open(f"{self.output_json_filepath}/{self.output_json_filename}.json", "r", encoding="utf-8") as json_file:
                predictions = list(json.load(json_file).values())
        
        for idx, entry in enumerate(tqdm(dataloader)):
            metadata = entry.pop("metadata")
            if not any(np.array([predictions[mdata["id"]] for mdata in metadata], dtype=str) == '0'):
                continue

            if not self.call_api:
                outputs = self.model.generate(input_ids=entry.input_ids,
                                            attention_mask=entry.attention_mask,
                                            eos_token_id=self.tokenizer.encode("</s>")[0],
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            # num_beams=2,
                                            # early_stopping=True,
                                            **self.generation_kwargs)
                print(f"outputs:\n{outputs}")
                prompt_len = int(entry.attention_mask.shape[1])
                outputs = outputs.tolist()
            else:
                input_texts = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True)
                outputs = api_get_tokens(self.api_name, input_texts, src_lang, tgt_lang, **self.chatgpt_kwargs)
                print(f"outputs:\n{outputs}")
            for mdata, output in zip(metadata, outputs):
                if not self.call_api:
                    generated = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                else:
                    generated = output
                # print("whole sentence:")
                # print(self.tokenizer.decode(output[:], skip_special_tokens=True))
                generated = generated.strip(self.tokenizer.pad_token).strip('\n').strip()
                if self.accelerator is not None:
                    process_data_idx.append(mdata['id'] * self.accelerator.num_processes + self.accelerator.process_index)
                    process_predictions.append(generated)
                else:
                    predictions[mdata["id"]]= generated
                print(f"generated:\n{generated}")
            if self.accelerator is not None:
                predictions_dict = {idx: prediction for idx, prediction in zip(process_data_idx, process_predictions)}
                with open(f"{self.output_json_filepath}/process{self.accelerator.process_index}_predictions.json", "w", encoding="utf-8") as json_file:
                    json.dump(predictions_dict, json_file, indent=4, ensure_ascii=False)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    for pid in range(self.accelerator.num_processes):
                        with open(f"{self.output_json_filepath}/process{pid}_predictions.json", "r", encoding="utf-8") as json_file:
                            p_dict = json.load(json_file)
                            for idx, prediction in p_dict.items():
                                predictions[int(idx)] = prediction
                                
            if self.accelerator is None or self.accelerator.is_main_process:
                predictions_dict = {idx: predictions[idx] for idx in range(len(predictions))}
                with open(f"{self.output_json_filepath}/{self.output_json_filename}.json", "w", encoding="utf-8") as json_file:
                    json.dump(predictions_dict, json_file, indent=4, ensure_ascii=False)               
        return predictions
