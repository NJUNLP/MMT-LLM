# ICL for Machine Translation
## Installation
使用库的所有内容，只需移动openicl文件夹至工作目录同级文件夹即可。（在此之前先安装好依赖）

安装依赖:
```
pip install -r requirements.txt
```

或者使用conda创建环境

```
conda create --name <env> --file requirements.txt
```

如果使用spBLEU metric，需要安装指定版本(https://github.com/facebookresearch/flores/tree/main/previous_releases/flores101):

```
git clone --single-branch --branch adding_spm_tokenized_bleu https://github.com/ngoyal2707/sacrebleu.git
cd sacrebleu
python setup.py install
```

## Usage

### OpenICL Framework

- In-context Learning Methods
  - 目前支持learning-free的部分retrieve方法，主要包括random, bm25, topk, votek。
- Inferencers
  - 目前支持的inference方法为计算ppl(用于解决NLU问题)以及生成式的inferencer(用于解决NLG问题)


- NLU Example

  以sst2数据集为例，使用topk retriever。
  ``` python
  from openicl import IclDatasetReader
  from openicl import IclTopkRetriever, IclPromptTemplate, IclPplInferencer
  from datasets import load_dataset
  
  def test_topk_sst2():
      ds = load_dataset('gpt3mix/sst2')
      dr = IclDatasetReader(ds, ctx_list=['text'], pred_label='label')  
      tp_dict = {0: "</E>Positive Movie Review: </Q>",
              1: "</E>Negative Movie Review: </Q>" }      
      tp = IclPromptTemplate(tp_dict, ctx_token_dict={'text':'</Q>'}, ice_token='</E>')
      rtr = IclTopkRetriever(dr, ice_template=tp, ice_num=8，batch_size=10)
      infr = IclPplInferencer(rtr, metric='acc', max_model_token_num=800)
      print(infr.score())
  ```
  步骤可归纳如下：

  1. **准备数据集**。一般是使用`datasets`库的`load_dataset`方法，从远程或者本地读取数据集，为`datasets.Dataset`或`datasets.DatasetDict`类型。(可以查看`datasets`库的[文档](https://huggingface.co/docs/datasets/v2.8.0/package_reference/loading_methods#datasets.load_dataset)，常用的数据集文件类型都是可以转换的) 样例中是直接从huggingface加载了数据集。
  2. **创建`IclDatasetReader`实例**。必需的参数包括：数据集、列表`ctx_list`， 以及`pred_label`。 `pred_label`为要预测的列的列名（字符串），`ctx_list`则应该包括所用用来预测`pred_label`列的列名（字符串列表）。 在没有提供模板类的情况下，`ctx_list`将决定retrieve时所使用的内容。(本样例中，`ctx_list`包含"text"列，所以Topk Retriever将使用"text"列去计算train set 和 validation set 的数据的embedding相似度)。
  3. (Optional) **创建`IclPromptTemplate`实例**。可以用在`IclDatasetReader`中，影响retrieve过程；也可以用在Retriever中，影响inference时最终prompt的形式。template可以为字典风格(方便解决分类问题，每个label对应一个不同的template)，也可以为字符串风格(方便解决生成类问题)。`ctx_token_dict`是一个映射字典，是列名到template中对应token的映射，用于后续data的替换。`ice_token`是可选项，是指生成prompt时，in-context examples 对应的特殊token，所以一般用于prompt template中。(本样例并没有采取分别声明`ice_template`和`prompt_template`的方式，而是只是用了一个`ice_template`。但由于声明了`ice_template`，它可以同时承担这两种模板的功能。在这两种prompt模板十分相像的情况下可以使用这种方式节省代码量)在test文件夹下，可以查看两种风格的template实例。(其中测试mtop、squad等都是使用的字符串风格模板)
  4. **创建`IclTopkRetriever`实例**。`ice_num`为in-context exmaple中包含的data条数。`batch_size`用于并行处理数据集。
  5. **创建`IclPplInferencer`实例**。指定metric(当前支持的metric还较少，但这里也可以直接是一个`IclBaseEvaluator`继承类)。`max_model_token_num`指定inference时模型可接受的最大tokenized长度，`model_name`和`tokenizer_name`可用来指定model和tokenizer(当前还只支持huggingface上的Pretrained Model, 后续会加入更多api支持)，这里都没有声明，默认是"gpt2-xl"。
  6. **调用score方法查看结果**。如只是想查看inference的结果，也可以直接调用`inference`方法。

### MT Test on Flores101 Dataset

使用默认参数对德语翻译英语的数据集进行测试，并计算BLEU score

```text
python test/test_flores101.py \
  --lang_pair deu-eng --retriever random --seed 43 --ice_num 8 \
  --prompt_template "</E></X>=</Y>" \
  --model_name facebook/xglm-7.5B --tokenizer_name facebook/xglm-7.5B \
  --output_dir demo --output_file deu-eng
```

- `lang_pair`: 测试语向
- `retriever`
  - retrieve方法，包括`random`, `bm25`, `topk`, `topk_mdl`, `votek`
  - 如果是`bm25`或者`topk`，可以选择`--oracle`实现oracle retrieve
- `seed`: 随机种子
- `ice_num`: in-context example个数
- `prompt_template`: 定义in-context example呈现和连接方式的模板
- `model_name`: 测试模型名或者本地路径
- `tokenizer_name`: 测试模型使用tokenizer名或者本地路径
- `output_dir`: 输出目录
- `output_file`: 输出文件

使用上面的参数可以大规模测试各语向，不同retrieve方法，不同ice数目，不同in-context template翻译的水平

`test_flores101.py`还可以实现下面的测试场景

#### Disorder Translation

```
python test/test_flores101.py \
  --lang_pair deu-eng --output_dir repeat --output_file deu-eng \
  --model_name facebook/xglm-7.5B --tokenizer_name facebook/xglm-7.5B \
  --disorder
```

打乱in-context examples中source sentence和target sentence的对应关系

#### Duplicated Translation

```
python test/test_flores101.py \
  --lang_pair deu-eng --output_dir repeat --output_file deu-eng \
  --model_name facebook/xglm-7.5B --tokenizer_name facebook/xglm-7.5B \
  --repeat
```

每组in-context examples都由第一个检索回的exmaple重复若干次组成

#### Mixed Translation Direction

```
python test/test_flores101.py \
  --lang_pair deu-eng --output_dir repeat --output_file deu-eng \
  --model_name facebook/xglm-7.5B --tokenizer_name facebook/xglm-7.5B \
  --reverse_direction \
  --direction_order 0 1 0 1 0 1 0 1
```

- `--reverse_direction`: 用于实现某些in-context exmaple翻译语向的翻转
- `--direction_order`: `ice_num`个$0/1$组成，$0$表示该example不翻转，$1$表示翻转

#### Cross Language ICE

```
python test/test_flores101.py \
  --lang_pair deu-eng --output_dir repeat --output_file deu-eng \
  --model_name facebook/xglm-7.5B --tokenizer_name facebook/xglm-7.5B \
  --cross_lang \
  --ex_lang rus fin tur \
  --lang_order 0 1 2 3 0 1 2 3
```

- `--cross_lang`: 用于实现不同源语言组成的in-context example
- `--ex_lang`: 列举用到的源语言
- `--lang_order`: 每个in-context example的源语言（原语向源语言用$0$表示，其他用`ex_lang`编号表示）

### Other Datasets

修改**test/config.py**中的内容

- `dataset_dir`: 数据集目录
- `lang_config`: 所有flores101语言
