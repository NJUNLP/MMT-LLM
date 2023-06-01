import sys
from os.path import dirname as d
from os.path import abspath, join
root = d(d(abspath(__file__)))
sys.path.append(root)

import argparse
from typing import List, Union, Optional
from copy import deepcopy
import numpy as np
import itertools

from openicl import IclDatasetReader
from openicl.icl_retriever import *
from openicl import IclPromptTemplate, IclGenInferencer, IclGenInferencer
from datasets import load_dataset, Dataset, DatasetDict

from config import dataset_dir

def load_dataset(src_lang, tgt_lang, args):
    valid_src = [line.strip() for line in open(join(dataset_dir, 'dev/{}.dev'.format(src_lang))).readlines()]
    valid_tgt = [line.strip() for line in open(join(dataset_dir, 'dev/{}.dev'.format(tgt_lang))).readlines()]
    test_src = [line.strip() for line in open(join(dataset_dir, 'devtest-first100/{}.devtest.first100'.format(src_lang))).readlines()]
    test_tgt = [line.strip() for line in open(join(dataset_dir, 'devtest-first100/{}.devtest.first100'.format(tgt_lang))).readlines()]
    if args.disorder:
        random.seed(args.seed)
        random.shuffle(valid_tgt)

    valid_dataset = Dataset.from_dict({'translation': [{src_lang: src, tgt_lang: tgt} for src, tgt in zip(valid_src, valid_tgt)], 
                                       src_lang: valid_src, 
                                       tgt_lang: valid_tgt})
    test_dataset = Dataset.from_dict({'translation': [{src_lang: src, tgt_lang: tgt} for src, tgt in zip(test_src, test_tgt)],
                                      src_lang: test_src,
                                      tgt_lang: test_tgt})
    return DatasetDict({'dev': valid_dataset, 'devtest': test_dataset})

def get_rtr(src, tgt, args):
    dr = IclDatasetReader(load_dataset(src, tgt, args), ctx_list=[src], pred_label=tgt)
    tp_str = args.prompt_template
    if "[source]" in tp_str:
        tp_str = tp_str.replace("[source]", src)
    if "[target]" in tp_str:
        tp_str = tp_str.replace("[target]", tgt)
    tp = IclPromptTemplate(
        tp_str, 
        ctx_token_dict={src: "</X>", tgt: "</Y>"},
        ice_token="</E>"
    )
    if args.retriever == "random":
        rtr = IclRandomRetriever(dr, ice_template=tp, ice_num=args.ice_num, select_split="dev", generation_split="devtest", seed=args.seed)
    elif args.retriever == "bm25":
        rtr = IclBM25Retriever(dr, ice_template=tp, ice_num=args.ice_num, select_split="dev", generation_split="devtest", oracle=args.oracle)
    elif args.retriever == "topk":
        rtr = IclTopkRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest", orcale=args.oracle)
    elif args.retriever == "topk_mdl":
        rtr = IclTopkMDLRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest")
    elif args.retriever == "votek":
        rtr = IclVotekRetriever(dr, ice_template=tp, ice_num=args.ice_num, batch_size=16, select_split="dev", generation_split="devtest")
    else:
        raise NotImplementedError
    return rtr

def repeat(ice_idx, rtr):
    assert len(ice_idx) > 0, len(ice_idx)
    ctx = ice_idx[0]
    for i in range(len(ice_idx)):
        ice_idx[i] = ctx

def test_flores(args):
    src, tgt = args.lang_pair.split('-')

    if args.reverse_direction:
        rtrs = [get_rtr(src, tgt, args), get_rtr(tgt, src, args)]
        rtr_order = args.direction_order
    elif args.cross_lang:
        rtrs = [rtr]
        for lang in args.ex_lang:
            rtrs.append(get_rtr(lang, tgt, args))
        rtr_order = args.lang_order
    else:
        rtrs = [get_rtr(src, tgt, args)]
        rtr_order = None
    infr = IclGenInferencer(
        rtrs, metric='bleu', max_model_token_num=1800, batch_size=8, rtr_order=rtr_order,
        ice_operator=repeat if args.repeat else None,
        model_name=args.model_name, tokenizer_name=args.tokenizer_name,
        output_json_filepath=args.output_dir,
        output_json_filename=args.output_file
    )
    score = infr.score(src_lang=src, tgt_lang=tgt)
    print("inference finished...")

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_pair", type=str, default=None)
    parser.add_argument("--retriever", type=str, default="random")
    parser.add_argument("--prompt_template", type=str, default="</E></X>=</Y>")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--ice_num", type=int, default=8)
    parser.add_argument("--oracle", default=False, action="store_true")

    parser.add_argument("--disorder", default=False, action="store_true")
    parser.add_argument("--repeat", default=False, action="store_true")

    parser.add_argument("--reverse_direction", default=False, action="store_true")
    parser.add_argument("--direction_order", nargs="+", type=int, default=None)

    parser.add_argument("--cross_lang", default=False, action="store_true")
    parser.add_argument("--ex_lang", nargs="+", type=str, default=None)
    parser.add_argument("--lang_order", nargs="+", type=int, default=None)

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    print(args)
    print(f"BLEU score = {test_flores(args)}")
