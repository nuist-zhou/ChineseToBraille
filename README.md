<h1 align="center">基于预训练语言模型与混合专家网络的中文到盲文翻译模型</h1>
</h1>

<p align="center">
陈大鹏，庄舟，申连顺，李晨凯 <br>
南京信息工程大学
</p>

---
<h2 align="center">摘  要</h2>
盲文作为视障群体获取信息的核心媒介，其与中文文本之间的精准转换是推动信息无障碍建设的关键技术环节。然而，传统翻译方法在处理中文多音字消歧与复杂分词规则时存在显著局限，难以满足实际应用中对高精度与高鲁棒性的需求。针对这一问题，提出了一种融合预训练语言模型与混合专家网络的中文到盲文翻译架构。该方法构建了端到端的编码器-解码器模型，其中编码器利用BERT获取丰富的深层语义表示，并引入混合专家网络机制以实现对不同语言现象的专家化建模，从而增强模型在复杂语义场景下的适应能力。为评估模型性能，构建了涵盖新闻、文学、日常交流等多领域的大规模中文-盲文平行语料库。实验结果显示，所提出的融合预训练语言模型与混合专家网络的中文到盲文翻译模型在测试集上取得了98.15%的BLEU分数，显著超越传统规则方法以及各类神经网络基线模型。进一步的消融实验表明，BERT语义表示与混合专家网络机制在提升翻译性能方面均发挥了关键作用。此外，通过用户研究获取的定性反馈表明，该方法能够有效提升视障用户的阅读流畅度与内容理解体验。

## 引言
BrailleTrans 是一种创新的中文到盲文翻译系统，它将预训练语言模型与混合专家网络相结合，以实现高精度的中文盲文转换。

模型框架如图所示：

![image](https://github.com/shen874/mwfy/raw/main/Framework.jpg)

## 快速入门
### BrailleTrans 的 Conda 环境
```bash
# Create and activate conda environment
conda create -n BrailleTrans python=3.10.18 -y
conda activate BrailleTrans
```

### 克隆项目
```bash
git clone https://github.com/CdpLab/BrailleTrans.git 
cd BrailleTrans
```

### 安装依赖
```bash
pip install -r requirements.txt
pip install torch==2.8.0 torchaudio==2.1.0
```
我们使用 `bert-base-chinese` 架构。由于该文件较大，请从[此处](https://huggingface.co/google-bert/bert-base-chinese)下载。然后将其放入 `/down` 文件夹中。文件结构如下：
```
·
├── down
·   ├── configs
    └── bert-base-chinese
```
### 数据准备
数据集放置在 `/data` 目录中。
```
·
├── data
·  
```
## 训练
使用以下命令开始训练你的 BrailleTrans 模型：
```
├── src
·   ├── train.py

python src/train.py \
--data_dir /path/to/your/data/ \          # Change to your data directory path
--bert_path /path/to/bert/model/ \        # Change to BERT model path
--epochs 10 \
--batch_size 32
```
## 结果
<table>
  <tr>
   <td><strong>Model</strong></td>  <td><strong>Train Loss</strong></td>
   <td><strong>Validation Loss</strong></td>  <td><strong>BLEU</strong></td>
  </tr>
  <tr><td>Transformer(Baseline)</td><td>0.057</td><td>0.042</td><td>86.51%</td></tr>
  <tr><td>+ MoE</td><td>0.056</td><td>0.040</td><td>92.12%</td></tr>
  <tr><td>+ BERT</td><td>0.038</td><td>0.031</td><td>97.02%</td></tr>
  <tr><td>+ BERT + MoE</td><td>0.033</td><td>0.032</td><td>98.15%</td></tr>
</table>
