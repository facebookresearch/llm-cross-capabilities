# Law of the Weakest Link: Cross Capabilities of Large Language Models
<p align="center">
  <a href="https://www.llm-cross-capabilities.org/"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
  <a href="https://arxiv.org/abs/2409.19951"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
  <a href="https://huggingface.co/datasets/MingZhong/crosseval" ><img src="https://img.shields.io/badge/🤗-CrossEval Benchmark-orange" height="25"></a>
</p>

🖋 **Authors:** [Ming Zhong](https://maszhongming.github.io/)\*, [Aston Zhang](https://www.astonzhang.com/)\*, [Xuewei Wang](https://www.linkedin.com/in/xuewei-wang-97a6b4190/), [Rui Hou](https://www.linkedin.com/in/rayhou/), [Wenhan Xiong](https://www.linkedin.com/in/wenhan-xiong-0a5984a3/), [Chenguang Zhu](https://cs.stanford.edu/~cgzhu/), [Zhengxing Chen](https://czxttkl.github.io/), [Liang Tan](https://www.linkedin.com/in/liang-tan-6646a484/), [Chloe Bi](https://www.linkedin.com/in/xueying-bi/), [Mike Lewis](https://ai.meta.com/people/209431298931133/mike-lewis/), [Sravya Popuri](https://www.linkedin.com/in/sravyapopuri/), [Sharan Narang](https://www.linkedin.com/in/sharan-narang/), [Melanie Kambadur](https://www.linkedin.com/in/melanie-kambadur/), [Dhruv Mahajan](https://www.linkedin.com/in/dhruv-mahajan-4397764/), [Sergey Edunov](https://www.linkedin.com/in/edunov/), [Jiawei Han](https://hanj.cs.illinois.edu/), [Laurens van der Maaten](https://lvdmaaten.github.io/)

## 📜 CrossEval: Benchmarking LLM Cross Capabilities

In real-world scenarios, many tasks require the intersection of multiple distinct capabilities across different types of expertise, which we refer to as **cross capabilities**. To explore this concept in the context of Large Language Models (LLMs), we present the **CrossEval**, a benchmark consisting of 1,400 expert-annotated prompts, 4,200 model-generated responses, and 8,400 human ratings with explanations. CrossEval is designed to evaluate the performance of LLMs across 14 capabilities, including:

### Single Capabilities
- English
- Reasoning
- Coding
- Image Recognition
- Tool Use
- Long Context
- Spanish

### Cross Capabilities
- Coding & Reasoning
- Image Recognition & Reasoning
- Tool Use & Coding
- Tool Use & Reasoning
- Long Context & Coding
- Spanish & Reasoning
- Spanish & Image Recognition

## 🛠️ Environment Setup
To get started, follow these steps to set up your Python environment:

```bash
conda create --name crosseval python=3.10
conda activate crosseval
pip install -r requirements.txt
```

## 📥 Loading the CrossEval Dataset

The CrossEval dataset is hosted on Hugging Face. You can load it as follows:

```python
from datasets import load_dataset

dataset = load_dataset("MingZhong/crosseval", split="test")
```

### Dataset Structure

Each instance in the dataset contains the following fields:

- **prompt_id**: Unique identifier for the prompt across capabilities
- **capability**: One of the 14 capabilities involved in the user prompt
- **difficulty**: Difficulty level of the prompt, categorized as 10% easy, 30% medium, 60% hard
- **l1_category**: High-level category for the user prompt
- **l2_category**: Subcategory for the user prompt
- **prompt**: The user-provided prompt text
- **attached_file**: URL of any attached file (used in image, long context, or tool use tasks)
- **response_i**: Model-generated responses (where `i=1,2,3` for multiple responses)
- **response_i_human_j_rating**: Human rating on a scale of 1-5 for each response (where `j=1,2` for multiple annotators)
- **response_i_human_j_explanation**: Human-provided explanations for the given rating

## 🚀 Generating Model Responses

To generate responses from different LLMs on the CrossEval dataset, ffollow the steps below.

### Setting API Keys
First, set up your API keys for the specified LLMs, ensuring compliance with the respective third-party terms of use.

```bash
export OPENAI_KEY="your_openai_api_key_here"          # GPT
export ANTHROPIC_API_KEY="your_claude_api_key_here"   # Claude
export GOOGLE_API_KEY="your_google_api_key_here"      # Gemini
```

For tool use prompts involving  code execution, responses may include generated files (e.g., plots). In these cases, the files are uploaded to a Hugging Face repository, and the URLs are included in the responses. Therefore, if you intend to run these prompts, you’ll also need to configure your Hugging Face API key:

```bash
export HF_KEY="your_huggingface_api_key_here"
```

Additionally, specify the account name and repository where the files will be saved at [this location](generate_response/models/gpt.py#L116).

### Generating Responses

Here’s an example script to generate responses using GPT-4o:

```bash
RESPONSE_DIR=outputs/responses
MODEL_NAME="gpt"
MODEL_VERSION="gpt-4o-2024-05-13"

python generate_response/generate.py \
    --save_path="${RESPONSE_DIR}/${MODEL_VERSION}.csv" \
    --model="${MODEL_NAME}" \
    --model_version="${MODEL_VERSION}" \
    --enable_code_interpreter \
```

Alternatively, you can execute the generation process using:

```bash
./scripts/get_response.sh
```

**Notes:**
- Update the `MODEL_NAME` and `MODEL_VERSION` in the script to match the specific model you want to evaluate.
- Model responses are saved as `{MODEL_VERSION}.csv` in the `RESPONSE_DIR` directory.
- The script supports resuming from the last processed instance if interrupted. Re-run the script to resume where it left off.

## 📊 Running Evaluations

To evaluate the generated responses, execute the following command:

```bash
MODEL_VERSION="gpt-4o-2024-05-13"
RESPONSE_DIR=outputs/responses
SCORE_DIR=outputs/scores
EVALUATOR=gpt

python evaluation/evaluate_response.py \
    --response=${MODEL_VERSION}_response \
    --response_file=${RESPONSE_DIR}/${MODEL_VERSION}.csv \
    --save_path=${SCORE_DIR}/${MODEL_VERSION}_response_${EVALUATOR}_score.csv \
    --evaluator=${EVALUATOR}
```

Alternatively, you can run:

```bash
./scripts/evaluate.sh
```

**Notes:**
- The script supports resuming from the last processed instance in case of an error. Simply re-run the script to continue the evaluation.
- The script will print the average scores for each capability after evaluation.
- Detailed scores for each prompt are saved in the `SCORE_DIR` directory.

## 🗂️ Model Outputs

We provide the responses and evaluations for the GPT and Llama model families on the CrossEval benchmark, available in [outputs/scores](outputs/scores) for reference.

Additionally, CrossEval is the largest meta-evaluation benchmark for examining correlations between LLM and human ratings. We release the LLM-generated ratings for reference responses in the [outputs/correlations](outputs/correlations) directory.

To compute correlation metrics between LLM and human ratings, run:

```bash
./scripts/get_correlation.sh
```

## License
The CrossEval benchmark is primarily intended to aid model research in the categorization, classification, or organization of data. This code and data is made available under a [CC-BY-NC license](https://creativecommons.org/licenses/by-nc/4.0/). However, you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models.
