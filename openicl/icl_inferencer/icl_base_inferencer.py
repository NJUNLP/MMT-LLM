'''Basic Inferencer'''
import os
import torch
from openicl import IclBaseRetriever
from openicl.utils.api_service import *
from openicl.icl_evaluator import *
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, GPT2Tokenizer
from typing import List, Union, Optional
from accelerate import Accelerator

class IclBaseInferencer:
    """Basic In-context Learning Inferencer Class
        Base class of In-context Learning Inferencer, no inference method
    Attributes:
        retriever: an IclBaseRetriever instance.
        metric: an IclBaseEvaluator instance.
        model: inferencer LM, could be initialized by name only or a config class. 
        tokenizer: a tokenizer initialized by AutoTokenizer.from_pretrained() method.
        max_model_token_num: LM maximum tokenized words allowed. 
        batch_size: int, batch size for DataLoader. 
    """
    model = None
    tokenizer = None
    call_api = False
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
        self.retriever = retriever
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.accelerator = accelerator
        self.api_name = api_name
        if references is None:
            self.references = self.retriever.generation_ds_references
        else:
            self.references = references
            if self.accelerator is not None:
                self.references = [self.references[idx] for idx in range(accelerator.process_index, len(self.references), accelerator.num_processes)]
        if isinstance(metric, str):
            metric = self._get_evaluator(metric)
        self.metric = metric
    
        self._get_api()
        if not self.call_api:
            self._get_model(self.model_name, model_config)
        self._get_tokenizer(self.tokenizer_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.model is not None:
            self.model.to(self.device)
        self.max_model_token_num = max_model_token_num
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        if not os.path.exists(self.output_json_filepath):
            os.makedirs(self.output_json_filepath)
            
    
    def inference(self) -> List:
        raise NotImplementedError("Method hasn't been implemented yet")
    
    
    def score(self, src_lang=None, tgt_lang=None):
        predictions = self.inference(src_lang, tgt_lang)
        if self.accelerator is None:
            return self.metric.score(predictions, src_lang=src_lang, tgt_lang=tgt_lang)
        if self.accelerator is not None and self.accelerator.is_main_process:
            return self.metric.score(predictions, src_lang=src_lang, tgt_lang=tgt_lang)
    
    
    def _get_evaluator(self, metric_str) -> IclBaseEvaluator:
        if metric_str == 'acc':
            return IclAccEvaluator(self.references)
        if metric_str == 'squad':
            return IclSquadEvaluator(self.references)
        if metric_str == 'bleu':
            return IclBleuEvaluator(self.references)
        raise ValueError(f"Invalid string {metric_str} for evaluator initialization")

    
    def _get_model(self, model_name, model_config):
        if model_config is not None:
            self.model = AutoModelForCausalLM.from_config(model_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)


    def _get_tokenizer(self, tokenizer_name):
        if self.api_name == 'opt-175b':
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
    
    def _get_api(self):
        if self.api_name == None:
            return
        self.call_api = is_api_available(self.api_name)
        if not self.call_api:
            UserWarning(f"api_name '{self.api_name}' is not available, Please check it")
    
        
    def get_input_token_num(self, inputs):
        return len(self.tokenizer(inputs, verbose=False)['input_ids'])
    
    