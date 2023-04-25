'''Prompt Template'''

from typing import Dict, Optional, List, Union
from .utils.check_type import _check_type_list, _check_dict

class IclPromptTemplate:
    """In-context Learning Prompt Template Class
        Provide prompt template for retriever to concatenate the text and generate in-context examples.
    Attributes:
        template: custom template dictionary or str. If dictionary: the key of the dictionary represents the value of pred_label, and value represents the corresponding generated statement. If str: representing a string of template 
        ctx_token_dict: dictionary mapping context column name to specific token. Token will be replaced by data in that column(one piece each time) in retrieve or inference process.
        val_mapping_dict: optional; value-to-value mapping dictionary used in self.template.
        mapping_field: optional; column name that self.val_mapping_dict will effect.
        ice_token: optional; str that represents the specific token mapping from in-context examples. None if you want retriever to use this template to generate in-context examples only, otherwise retriever could use this template to generate the final prompt when inferencing as well(ice + pred_data_instruction). ice_token would be invisible when generating in-context examples.
    """
    def __init__(self,
                 template: Union[Dict, str],
                 ctx_token_dict: Dict,
                 val_mapping_dict: Optional[Dict] = None,
                 mapping_field: Optional[str] = None,
                 ice_token: Optional[str] = None
    ) -> None:
        self.template = _check_type_list(template, [Dict, str])
        self.ctx_token_dict = _check_dict(ctx_token_dict)
        self.val_mapping_dict = _check_type_list(val_mapping_dict, [None, Dict])
        self.mapping_field = _check_type_list(mapping_field, [None, str])
        self.ice_token = _check_type_list(ice_token, [None, str])
        if (self.mapping_field is not None and self.val_mapping_dict is None) or \
        self.mapping_field is None and self.val_mapping_dict is not None:
            raise ValueError("self.mapping_field and self.val_mapping_dict should be set together")
        self.check_token_dict_legacy()
        

    def check_token_dict_legacy(self):
        if isinstance(self.template, Dict):
            # check if token exists in values of tp_dict 
            for tp_dict_val in self.template.values():
                if not isinstance(tp_dict_val, str):
                    raise TypeError(f"tp_dict expected a str value, but got '{tp_dict_val}'")
                for ctx_token_val in self.ctx_token_dict.values():
                    if ctx_token_val not in tp_dict_val:
                        raise LookupError(f"'{ctx_token_val}' not in '{tp_dict_val}'")
                if self.ice_token is not None and self.ice_token not in tp_dict_val:
                    raise LookupError(f"'{self.ice_token}' not in '{tp_dict_val}'")
        if isinstance(self.template, str):
            for ctx_token_val in self.ctx_token_dict.values():
                if ctx_token_val not in self.template:
                    raise LookupError(f"'{ctx_token_val}' not in '{self.template}'")
            if self.ice_token is not None and self.ice_token not in self.template:
                raise LookupError(f"'{self.ice_token}' not in '{self.template}'")
                    
        # check duplicates
        if len(self.ctx_token_dict.values()) != len(set(self.ctx_token_dict.values())):
            raise ValueError(f"There are duplicates in self.ctx_token_dict.values()")
        if self.ice_token is not None and self.ice_token in self.ctx_token_dict.values():
            raise ValueError(f"There are duplicates between self.ctx_token_dict.values() and self.ice_token")
                

    def generate_ice_item(self, entry, pred_label):
        # select the corresponding template 
        tp = self.template[pred_label] if isinstance(self.template, Dict) else self.template
        # remove ice_token
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, '')
        # replace context token
        for key, token in self.ctx_token_dict.items():
            if self.val_mapping_dict is not None and key == self.mapping_field:
                tp = tp.replace(token, str(self.val_mapping_dict[pred_label]))
            else: 
                tp = tp.replace(token, str(entry[key]))
        return tp
    
    
    def generate_label_prompt_item(self, entry, ice, pred_label):
        if self.ice_token is None:
            raise ValueError("IctPromptTemplate.ice_token should be not None when generate prompt")
        # select the corresponding template
        tp = self.template[pred_label] if isinstance(self.template, Dict) else self.template
        # insert in-context examples
        tp = tp.replace(self.ice_token, ice)
        # replace context token
        for key, token in self.ctx_token_dict.items():
            if self.val_mapping_dict is not None and key == self.mapping_field:
                tp = tp.replace(token, str(self.val_mapping_dict[pred_label]))
            else: 
                tp = tp.replace(token, str(entry[key]))
        return tp        
    
    
    def generate_item(self, entry, gen_field=None, gen_field_replace_token='', ice_field_replace_token=''):
        tp = None
        if isinstance(self.template, str):
            tp = self.template
        else:
            pred_label = None
            if self.mapping_field is not None:
                pred_label = entry[self.mapping_field]
            if pred_label in self.template.keys():
                tp = self.template[pred_label]
            else:
                tp = self.template[list(self.template.keys())[0]]
        if self.ice_token is not None:
            tp = tp.replace(self.ice_token, ice_field_replace_token)
        for key, token in self.ctx_token_dict.items():
                if gen_field is not None and key == gen_field:
                    tp = tp.replace(token, gen_field_replace_token)
                else:
                    tp = tp.replace(token, str(entry[key]))
        return tp
        
    
    def _check_prompt_template(obj) -> "IclPromptTemplate":
        if isinstance(obj, IclPromptTemplate):
            return obj
        else:
            raise TypeError(f"Expected a IclPromptTemplate object, but got {obj}") 
        
        
    def __repr__(self):    
        return f"IclPromptTemplate({{\n    tp_dict: {self.template},\n    ctx_token_dict: {self.ctx_token_dict},\n    ice_token: {self.ice_token}\n}})"
    