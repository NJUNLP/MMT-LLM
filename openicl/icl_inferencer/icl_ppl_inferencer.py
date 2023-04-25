'''PPL Inferencer'''

import json
import torch
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer import IclBaseInferencer
from openicl.utils.api_service import *
from typing import List, Union, Optional
from tqdm import tqdm
from tqdm import trange
from transformers import PretrainedConfig
from accelerate import Accelerator

class IclPplInferencer(IclBaseInferencer):
    def __init__(self,
                 retriever: IclBaseRetriever,
                 metric: Optional[Union[str, IclBaseEvaluator]] = 'acc',
                 references: Optional[List] = None,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 api_name: Optional[str] = None
    ) -> None:
        super().__init__(retriever, metric, references, model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, api_name)


    def inference(self, src_lang, tgt_lang):
        tot_num = len(self.retriever.generation_ds)
        if self.accelerator is not None:
            tot_num *= int(self.accelerator.num_processes)
        predictions = [0 for _ in range(tot_num)]
        sub_predictions = []
        ppl = []
        ice = []
        ice_idx_list = self.retriever.retrieve()
        labels = self.retriever.get_labels()

        for idx in trange(len(ice_idx_list)):
            ice.append(self.retriever.generate_ice(ice_idx_list[idx]))

        for label in labels:
            prompt_list = []
            sub_ppl_list = []
            for idx in range(len(ice_idx_list)):
                prompt = self.retriever.generate_label_prompt(idx, ice[idx], label)
                if self.max_model_token_num is not None:
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                            ice_idx_list[idx] = ice_idx_list[idx][:-1]
                            ice[idx] = self.retriever.generate_ice(ice_idx_list[idx])
                            prompt = self.retriever.generate_label_prompt(idx, ice[idx], label)
                            prompt_token_num = self.get_input_token_num(prompt)
                prompt_list.append(prompt)

            for idx in trange(0, len(prompt_list), self.batch_size):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                with torch.no_grad():
                    sub_res = self.cal_ppl(sub_prompt_list).tolist()
                for res in sub_res:
                    sub_ppl_list.append(res)

            ppl.append(sub_ppl_list)

        ppl = list(zip(*ppl))
        for single_ppl in ppl:
             sub_predictions.append(labels[single_ppl.index(min(single_ppl))])

        if self.accelerator is not None:
            predictions_dict = {idx * self.accelerator.num_processes + self.accelerator.process_index: sub_predictions[idx] for idx in range(len(sub_predictions))}
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
            with open(f"{self.output_json_filepath}/predictions.json", "w", encoding="utf-8") as json_file:
                json.dump(predictions_dict, json_file, indent=4, ensure_ascii=False)
        return predictions


    def cal_ppl(self, input_texts: List[str]):
        if self.call_api:
            return api_get_ppl(self.api_name, input_texts)
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
