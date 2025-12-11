<div align="center">

<h1> GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

<h5 align="center"> 

<a href='https://www.lanxplanet.com/GEM-ECG/'><img src='https://img.shields.io/badge/Homepage-8A2BE2'></a>
<a href='https://arxiv.org/pdf/2503.06073'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/LANSG/GEM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://huggingface.co/datasets/LANSG/ECG-Grounding'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>


[Xiang Lan](https://www.lanxplanet.com/)<sup>1</sup>,
[Feng Wu](https://meiyoufeng116.github.io)<sup>1</sup>,
[Kai He](https://kaihe-better.github.io)<sup>1</sup>,
[Qinghao Zhao]()<sup>2</sup>,
[Shenda Hong](https://hsd1503.github.io/)<sup>3</sup>,
[Mengling Feng](https://www.mornin-feng.com/me)<sup>1</sup>

<sup>1</sup>[National University of Singapore](https://nus.edu.sg/) <sup>2</sup>[Peking University People's Hospital](https://english.pkuph.cn/) <sup>3</sup>[Peking University](https://english.pku.edu.cn/)


<p align="center">
    <img src="pics/fig1_v.png" width="90%">
</p>

</h5>
</div>

## Introduction

While recent multimodal large language models (MLLMs) have advanced automated ECG interpretation, they still face two key limitations: (1) insufficient multimodal synergy between ECG time series signals and ECG images, and (2) limited explainability in linking diagnoses to granular waveform evidence. We introduce GEM, the first MLLM unifying ECG time series, 12-lead ECG images and text for grounded and clinician-aligned ECG interpretation. GEM enables feature-grounded analysis, evidence-driven reasoning, and a clinician-like diagnostic process through three core innovations: a dual-encoder framework extracting complementary time series and image features, cross-modal alignment for effective multimodal understanding, and knowledge-guided instruction data generation for generating high-granularity grounding data (ECG-Grounding) linking diagnoses to measurable parameters (e.g., QRS/PR Intervals). Additionally, we propose the Grounded ECG Understanding task, a clinically motivated benchmark designed to comprehensively assess the MLLM's capability in grounded ECG understanding. Experimental results on both existing and our proposed benchmarks show GEM significantly improves predictive performance (CSN +7.4%↑), explainability (+22.7%↑), and grounding (+25.3%↑), making it a promising approach for real-world clinical applications.

## 🔥Updates

- **[Sep 2025]** GEM has been accepted to NeurIPS 2025! More updates coming soon.
- **[Jul 2025]** The full version of MIMIC-IV-ECG with beat-level features and GPT-4o interpretations has been released — check it out [here](https://arxiv.org/pdf/2507.15255)!
- **[Mar 2025]** GEM-7B and ECG-Grounding-30k are now available. 

We will continue to release more ECG-Grounding data and associated beat-level features progressively. 

*Stay tuned for updates!*

## Resource

#### Project Page: 📖 [Page](https://www.lanxplanet.com/GEM-ECG/)

#### Paper: 📄 [Arxiv](https://arxiv.org/pdf/2503.06073)

#### Model: 🤗 [GEM](https://huggingface.co/LANSG/GEM)

#### Data: 🤗 [ECG-Grounding](https://huggingface.co/datasets/LANSG/ECG-Grounding)

## Setup

```shell
git clone https://github.com/lanxiang1017/GEM.git
bash GEM/setup.sh
```

## Data Preparation

Please download required data:

ECG:  
- [MIMIC-IV](https://physionet.org/content/mimic-iv-ecg/1.0/)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [Code-15%](https://zenodo.org/records/4916206)
- [CPSC 2018](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/)
- [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- [G12E](https://physionet.org/content/challenge-2020/1.0.2/training/georgia/)

Images:
- [ECG-Grounding-Images](https://huggingface.co/datasets/LANSG/ECG-Grounding) (mimic_gen)
- [PTB-XL-Test-Images](https://huggingface.co/datasets/LANSG/ECG-Grounding) (ptb-xl-gen)
- [ECG-Instruct](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/tree/main)
- [ECG-Bench](https://huggingface.co/datasets/PULSE-ECG/ECGBench)

After downloading all of them, organize the data as follows in `./data`,

```
├── ecg_timeseries
    └── champan-shaoxing
    └── code15
    └── cpsc2018
    └── ptbxl
    └── georgia
    └── mimic-iv
├── ecg_images
    └── cod15_v4
    └── csn_aug_all_layout_papersize
    └── csn_ori_layout_papersize
    └── csn_part_noise_layout_papersize
    └── gen_images
      └── mimic_gen
      └── ptb-xl-gen
    └── mimic
    └── mimic_v4
    └── ptb-xl
├── ecg_bench
    └── images
    └── ecg-grounding-test-mimiciv.json
    └── ecg-grounding-test-ptbxl.json
├── ecg_jsons
    └── ECG_Grounding_30k.json

```

## Pretrained Model Preparation

Pretrained ECG Encoder:
  - [ECG-CoCa](https://drive.google.com/drive/folders/1-0lRJy7PAMZ7bflbOszwhy3_ZwfTlGYB?usp=sharing) : download ```cpt_wfep_epoch_20.pt```, place it in ```GEM/ecg_coca/open_clip/checkpoint```

Pretrained MLLMs:
  - [PULSE](https://huggingface.co/PULSE-ECG/PULSE-7B)  
  - [LLaVA](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)

## Train

For training from scratch:
  - step 1. specify paths in ```GEM/scripts/train_gem.sh```
  - step 2. run ```bash GEM/scripts/train_gem.sh```

## Evaluation

For ECG-Grounding:
  - step 1. generate interpretations: ```GEM/evaluation/gem_bench/bench_ecggrounding.sh```
  - step 2. process interpretations: ```GEM/gem_evaluation/process_gem_outputs.ipynb```
  - step 3. generate GPT evaluation reports: ```GEM/gem_evaluation/generate_gpt_eval.py```
  - step 4. process evaluation reports and get scores: ```GEM/gem_evaluation/process_grounding_scores.ipynb```

For ECG-Bench:
  - step 1. generate results: ```GEM/evaluation/gem_bench/bench_ecgbench.sh```
  - step 2. evaluate results: ```GEM/evaluation/evaluate_ecgbench.py```
  - step 3. evaluate reports: ```GEM/evaluation/eval_report.py```

*Note*
- 1. You need to specify the result paths in all evaluation scripts (For ECG-Bench, you also need to specify the path to question files in evaluate_ecgbench.py).
- 2. If you download our trained GEM-7B model from HuggingFace, you must set the path to ECG-CoCa in the config.json file (under "mm_ecg_tower") before using it.
- 3. bench_ecggrounding.sh is designed to use multiple GPUs to generate interpretations simultaneously, reducing generation time. To use it, you must split the test file (ecg-grounding-test-mimiciv.json) into multiple chunks. If you prefer a simpler setup, you can use bench_ecgbench.sh instead. The core generation functions are the same. Example usage: ```bash bench_ecgbench.sh -m PATH_TO_GEM -d ecg-grounding-test-mimiciv```.
 

## Citation

If you find GEM helpful for your research and applications, please cite our paper:

```bibtex
@misc{lan2025gemempoweringmllmgrounded,
      title={GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images}, 
      author={Xiang Lan and Feng Wu and Kai He and Qinghao Zhao and Shenda Hong and Mengling Feng},
      year={2025},
      eprint={2503.06073},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.06073}, 
}
```

## Acknowledgement
We thank the authors of [PULSE](https://github.com/AIMedLab/PULSE/tree/dev) and [ECG-Chat](https://github.com/YubaoZhao/ECG-Chat) for their publicly released models, datasets, and training codes.

