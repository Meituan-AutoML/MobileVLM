# MobileLLaMA SFT

## 🛠️ Installation
Our MobileLLaMA SFT training code is based on [FastChat](https://github.com/lm-sys/FastChat) (commit id: 81785d7ed1d6afb966b464a8ee4689b7413e6313)

### Install From Source.
1. Clone the [FastChat](https://github.com/lm-sys/FastChat) repository and navigate to the FastChat folder
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```
If you are running on Mac:
```bash
brew install rust cmake
```
2. Install package
```bash
pip3 install --upgrade pip
pip3 install -e ".[model_worker,webui]"
```
## Model Weights

You can download MobileLLaMA-1.4B-Base / MobileLLaMA-2.7B-Base model from huggingface website to your local path, or run our train.sh directly to download the weights before training:
- [MobileLLaMA-1.4B-Base](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base)
- [MobileLLaMA-2.7B-Base](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Base)

## Dataset
We use the sft dataset in Vicuna fromat can be download from link: [ShareGPT_Vicuna_dataset](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered), and follow the steps:
1. download the [json](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V4.3_unfiltered_cleaned_split.json) file to local data path.
2. write the correct "--data_path" in your SFT training scripts.

## 💎 Training
Our training process can be reproduced by runing the scrips:
```bash
cd MobileVLM
# for MobileLLaMA-1.4B
sh mobilellama/sft/sft_MobileLLaMA-1.4B-Base.sh
# for MobileLLaMA-2.7B
sh mobilellama/sft/sft_MobileLLaMA-2.7B-Base.sh
```
Weights after SFT training can be download from:
- [MobileLLaMA-1.4B-Chat](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Chat)
- [MobileLLaMA-2.7B-Chat](https://huggingface.co/mtgv/MobileLLaMA-2.7B-Chat)
## Evaluation results
The performance comparison of the model on several benchmarks before and after Supervised Fine-Tuning (SFT), as illustrated below:

<table>
    <tr>
      <th style="text-align: center;">models</th>
      <th style="text-align: center;" colspan="2">knowledge</th>
      <th style="text-align: center;">reasoning</th>
      <th style="text-align: center;" colspan="3">Understanding</th>
    </tr>
    <tr>
      <td style="text-align: center;">tasks</td>
      <td style="text-align: center;">TriviaQA</td>
      <td style="text-align: center;">NQ</td>
      <td style="text-align: center;">HellaSwag</td>
      <td style="text-align: center;">RACEMiddle</td>
      <td style="text-align: center;">RACEHigh</td>
      <td style="text-align: center;">XSum</td>
    </tr>
    <tr>
        <td style="text-align: center;">MobileLLaMA 1.4B Base</td>
        <td style="text-align: center;">15.7</td>
        <td style="text-align: center;">2.9</td>
        <td style="text-align: center;">43.0</td>
        <td style="text-align: center;">21.5</td>
        <td style="text-align: center;">22.7</td>
        <td style="text-align: center;">18.0</td>
    </tr>
    <tr>
        <td style="text-align: center;">MobileLLaMA 1.4B sft</td>
        <td style="text-align: center;">20.3</td>
        <td style="text-align: center;">3.9</td>
        <td style="text-align: center;">45.0</td>
        <td style="text-align: center;">25.7</td>
        <td style="text-align: center;">26.6</td>
        <td style="text-align: center;">20.7</td>
    </tr>
    <tr>
      <td style="text-align: center;">MobileLLaMA 2.7B Base</td>
      <td style="text-align: center;">23.0</td>
      <td style="text-align: center;">4.2</td>
      <td style="text-align: center;">48.0</td>
      <td style="text-align: center;">23.8</td>
      <td style="text-align: center;">24.6</td>
      <td style="text-align: center;">16.8</td>
    </tr>
    <tr>
        <td style="text-align: center;">MobileLLaMA 2.7B sft</td>
        <td style="text-align: center;">26.4</td>
        <td style="text-align: center;">8.3</td>
        <td style="text-align: center;">50.0</td>
        <td style="text-align: center;">26.7</td>
        <td style="text-align: center;">27.2</td>
        <td style="text-align: center;">23.8</td>
    </tr>
  </table>


## 🤝 Acknowledgments
- [Vicuna](https://github.com/lm-sys/FastChat): the SFT codebase we utilize. Thanks for their wonderful work! 👏
- [ShareGPT_Vicuna_dataset](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered): the dataset we train our chat model, Thanks for their well collection! 👏!
