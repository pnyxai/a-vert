# A-VERT

A-VERT is a method for comparing LM generations to target responses. It is intended to replace the `exact-match` or `logprobs` technique normally used in benchmarks, which makes evaluations diverge from real-world scenarios.

This repository is ordered as follows:
- `./a_vert` : Code for the `a_vert` library.
- `./lm-eval_tasks` : [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) compatible tasks that use `a_vert` library.
- `./notebooks` : Ipython notebooks used to produce the A-VERT paper results.
- `./testing` : Helper code used to load `lm-eval` results but not used in the `a_vert` library.



### Installing

The package is available on pip (soon):

```sh
pip install a_vert
```

### Building

We use poetry to manage the package, to install just do:

```sh
poetry install
```

### Usage

In order to use `a_vert` you need to have an embeddings or reranker model deployed and the access data available in the following environment variables:
- `AVERT_MODEL_ENDPOINT` : Endpoint of the embedding or reranker model, for example: `http://127.0.0.1:8000`.
- `AVERT_ENDPOINT_TYPE` : Should be `vllm` or `tei` depending on the backend used in the endpoint. The `vllm` type is openai-API compatible.
- `AVERT_MODEL_NAME` : The name that you give your model, for the example below, this should be `pocket_network`.

To run a test task (2 examples, fast), just use the `lm-eval` library and add the provided tasks as additional tasks: 

```sh
lm_eval \
    --model local-chat-completions \
    --tasks babi-task_01-single_supporting_fact \
    --model_args '{"base_url":"https://your.favorite.endpoint/v1/chat/completions","timeout":"600","max_retries":3,"tokenized_requests":false, "model":"your_favorite_model"}' \
    --num_fewshot 0 \
    --apply_chat_template \
    --trust_remote_code \
    --include_path ./lm-eval_tasks \
    --limit 2
```

### Model deployment example

We recommend to use rerankers from the Qwen3 family, depending on the available hardware, use [tomaarsen/Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) or [tomaarsen/Qwen3-Reranker-4B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-4B-seq-cls). 
To deploy the model you can use [vLLM](https://github.com/vllm-project/vllm) and `docker-compose`, here is a sample yaml:

```yaml
version: '3'
services:
  vllm-openai-embeddings:
    container_name: vllm-openai-embeddings
    image: vllm/vllm-openai:v0.10.0
    volumes:
      - ${MODELS_PATH}:/root/.cache/huggingface/hub/
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - MODEL_NAME=${MODEL_NAME}
      - NUM_GPUS=${NUM_GPUS}
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
      - MAX_MODEL_LEN=${MAX_MODEL_LEN}
      - MAX_NUM_SEQS=${MAX_NUM_SEQS}
      - SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
      - TASK=${TASK}
    entrypoint: ["python3",
      "-m",
      "vllm.entrypoints.openai.api_server",
      "--task",
      "${TASK}",
      "--model",
      "${MODEL_NAME}",
      "--served-model-name",
      "${SERVED_MODEL_NAME}",
      "--tensor-parallel-size",
      "${NUM_GPUS}",
      "--gpu-memory-utilization",
      "${GPU_MEMORY_UTILIZATION}",
      "--max-model-len",
      "${MAX_MODEL_LEN}",
      "--trust-remote-code",
      "--max-num-seqs",
      "${MAX_NUM_SEQS}",
      "--max-num-batched-tokens",
      "${MAX_MODEL_LEN}",
      ]
    ports:
     - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
```
with the following `.env` file (in the same directory as the `docker-compose.yaml`):
```envfile
HF_TOKEN=hf_lalala
MODELS_PATH=/media/some_disk/where_you_store_models/
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=14000
MAX_NUM_SEQS=32
NUM_GPUS=1
TASK=classify
MODEL_NAME=/root/.cache/huggingface/hub/Qwen3-Reranker-4B-seq-cls
SERVED_MODEL_NAME=pocket_network
```


# Paper

The paper is available [here](). 
In order to reproduce the results, please [download the test data from here (~480 MB)](https://drive.google.com/file/d/1lMh5-MWtOKGw4j9-MEC7Vlywge--DG0t/view?usp=sharing) and extract it in the root of this repository (approx 3.1 GB of space is needed). The notebooks will look for it at: `./data`.

Abstract:
> The automatic evaluation of Language Model (LM) responses is a critical piece in the development of benchmarks and metrics, both for model training and quality assessment of production model endpoints. The current approaches to response classification relies on methods that are too expensive (i.e. LLM-as-a-Judge) or that are far from real-world conditions (string-matching, logprob). In this paper, a structure-free evaluation method is presented. The method makes use of semantic embedding distances to match target candidates with arbitrary LM-generated text, resulting in a robust classification of the response at a relatively low compute cost (embedding models of less than 10B parameters). The results show a regression score of ~0.97 and an accuracy of ~96% against human annotators, tested over 3 data sets and 3 different LM architectures.

citation:
```bibtext
TODO
```