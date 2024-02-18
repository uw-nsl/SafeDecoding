# SafeDecoding

This is the official repository for "[SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding](https://arxiv.org/abs/2402.08983)

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2402.08983)

## Abstract

As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

## Dataset
This dataset contains attack prompts generated from GCG, AutoDAN, PAIR, and DeepInception for **research use ONLY**.
[Huggingface](https://huggingface.co/datasets/flydust/SafeDecoding-Attackers)

## Getting Start
**Get Code**
```
cd /
git clone https://github.com/uw-nsl/SafeDecoding.git
```
**Build Environment**
```
cd SafeDecoding
conda create -n SafeDecoding python=3.10
conda activate SafeDecoding
pip install -r requirements.txt
```
**[Optional] Get access to attack dataset and Llama2-chat model from Huggingface**

Before doing so, make sure you have premission to the dataset and the llama2 model.
```
huggingface-cli login
```
then enter your Huggingface private key begin with "hf_".

## Acknowledgements

Some codes are build upon [PEFT](https://github.com/huggingface/peft), [llm-attack](https://github.com/llm-attacks/llm-attacks), [BPE-Dropout](# https://github.com/VProv/BPE-Dropout/), [lmppl](https://github.com/asahi417/lmppl/).

## Citation
```
@misc{xu2024safedecoding,
      title={SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding}, 
      author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Jinyuan Jia and Bill Yuchen Lin and Radha Poovendran},
      year={2024},
      eprint={2402.08983},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```