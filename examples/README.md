# A-VERT Docker Deployment Examples

This directory contains example configurations for deploying A-VERT models using Docker Compose. The setup includes both an A-VERT model (for embeddings/reranking) and an LLM model served via vLLM.

## Overview

The `docker-compose.yaml` file defines two services:
- **vllm-openai-avert**: Serves an embedding or reranker model for A-VERT evaluation
- **vllm-openai-llm**: Serves a Large Language Model for generating responses

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured
- At least one GPU available
- Models pre-downloaded to your a local directory.

## Setup

### 1. Configure Environment Variables

Copy the `example.env` file to `.env` and update the values:

```bash
cp example.env .env
```

Edit `.env` with your configuration:

```bash
# General Configuration
HF_TOKEN=hf_your_actual_token_here
MODELS_PATH=/path/to/your/models
NUM_GPUS=1

# A-VERT Model Configuration
AVERT_GPU_IDS=0                                          # GPU ID(s) for A-VERT
TASK=classify                                            # Task type: embed, score, or classify
AVERT_MODEL_NAME=tomaarsen/Qwen3-Reranker-4B-seq-cls    # Model to use following the Hugging Face naming (org/repo)
AVERT_SERVED_MODEL_NAME=avert-model                     # Name for API requests
AVERT_GPU_MEMORY_UTILIZATION=0.5                        # GPU memory fraction
AVERT_MAX_MODEL_LEN=8192                                # Max context length
AVERT_MAX_NUM_SEQS=256                                  # Max parallel sequences

# LLM Configuration
LLM_GPU_IDS=0                                           # GPU ID(s) for LLM
LLM_MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct        # LLM model to use following the Hugging Face naming (org/repo)
LLM_SERVED_MODEL_NAME=llm-model                         # Name for API requests
LLM_GPU_MEMORY_UTILIZATION=0.9                          # GPU memory fraction
LLM_MAX_MODEL_LEN=8192                                  # Max context length
LLM_MAX_NUM_SEQS=256                                    # Max parallel sequences
HF_HUB_OFFLINE=1                                        # 1: offline only (default), 0: allow downloads
```

**Note**: By default, `HF_HUB_OFFLINE=1` is set, which means Docker containers will only use pre-downloaded models. Make sure your models are downloaded to `MODELS_PATH` before starting the containers.

> Model Download
> 
> Before running the containers, you need to download the models to a local directory. 
> 
> Make sure the `MODELS_PATH` in your `.env` file matches the local directory used above.

### 2. Recommended Models

We recommend using rerankers from the Qwen3 family for A-VERT:

- **Small**: [tomaarsen/Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls)
- **Medium**: [tomaarsen/Qwen3-Reranker-4B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-4B-seq-cls)
- **Large**: [tomaarsen/Qwen3-Reranker-8B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-8B-seq-cls)

## Running the Services

### Start Both Services

```bash
docker-compose up -d vllm-openai-avert vllm-openai-llm
```

## API Endpoints

Once running, the services expose OpenAI-compatible APIs:

- **A-VERT Model**: `http://localhost:8000`
- **LLM Model**: `http://localhost:8001`

### Testing A-VERT Endpoint

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "avert-model",
    "input": [
      ["What is the capital of France?", "Paris is the capital of France."],
      ["What is the capital of France?", "London is the capital of England."]
    ]
  }'
```

### Testing LLM Endpoint

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-model",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Congratulations! You now have an LLM ready to be evaluated with the `A-VERT` methodology.
Please follow the instructions from the [A-VERT README](../README.md) to run evaluations using `lm-eval-harness` package.


## Using with A-VERT Library

After deploying the A-VERT model, set these environment variables to use it:

```bash
export AVERT_MODEL_ENDPOINT="http://localhost:8000"
export AVERT_ENDPOINT_TYPE="vllm"
export AVERT_MODEL_NAME="avert-model"  # Must match AVERT_SERVED_MODEL_NAME
export AVERT_METHOD="rerank"
export AVERT_PROMPT_TEMPLATE="qwen3-reranker"
```

Then run your A-VERT evaluations (be sure to have `a_vert` and `lm-eval` installed at this point):

```bash
lm_eval \
    --model local-chat-completions \
    --tasks babi-task_01-single_supporting_fact \
    --model_args '{"base_url":"http://localhost:8001/v1/chat/completions","timeout":"600","max_retries":3,"tokenized_requests":false, "model":"llm-model"}' \
    --num_fewshot 0 \
    --apply_chat_template \
    --trust_remote_code \
    --include_path ../lm-eval_tasks \
    --limit 2
```


## Advanced GPU Configuration

### Using Different GPUs for Each Service

If you have multiple GPUs, you can run each service on a separate GPU:

```bash
# In .env
AVERT_GPU_IDS=0
LLM_GPU_IDS=1
```

### Using Multiple GPUs for One Service

For tensor parallelism (requires updating NUM_GPUS):

```bash
# In .env
NUM_GPUS=2
AVERT_GPU_IDS=0,1
LLM_GPU_IDS=2,3
```

### Sharing a GPU (if memory allows)

Both services can run on the same GPU if memory permits:

```bash
# In .env
AVERT_GPU_IDS=0
LLM_GPU_IDS=0
AVERT_GPU_MEMORY_UTILIZATION=0.3
LLM_GPU_MEMORY_UTILIZATION=0.6
```