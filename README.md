## Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis
### Requirement
Our code is based on [OpenICL](https://arxiv.org/abs/2303.02913) framework.
More details and guidance can be found in this repository: https://github.com/Shark-NLP/OpenICL.

### Evaluation 

#### Dataset
We evaluate large language model's multilingual translation abilities on Flores-101 dataset, which can be downloaded with this [link](https://github.com/facebookresearch/flores/blob/main/previous_releases/flores101/README.md).

#### Scripts
Below is our evaluation script.
```shell
python test/test_flores101.py \
  --lang_pair deu-eng \
  --retriever random \
  --ice_num 8 \
  --prompt_template "</E></X>=</Y>" \
  --model_name your-model-name 
  --tokenizer_name your-tokenizer-name \
  --output_dir your-output-path \
  --output_file your-output-file \
  --seed 43
```

### Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@misc{zhu2023multilingual,
      title={Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis}, 
      author={Wenhao Zhu and Hongyi Liu and Qingxiu Dong and Jingjing Xu and Shujian Huang and Lingpeng Kong and Jiajun Chen and Lei Li},
      year={2023},
      eprint={2304.04675},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```