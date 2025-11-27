# Sample Tasks for `lm-eval`

This folder contains tasks in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) format that implement `a-vert` as a metric.

Please make sure to have deployed an embedding/reranker model endpoint (we recommend the `Qwen3-reranker` series), and configure the following environment variables:

```sh
export AVERT_MODEL_ENDPOINT="http://localhost:8000"
export AVERT_ENDPOINT_TYPE="vllm"
export AVERT_MODEL_NAME="avert-model"
export AVERT_METHOD="rerank"
export AVERT_PROMPT_TEMPLATE="qwen3-reranker"
```

Please refer to [the example README](../examples/README.md) for more details on how to deploy using `docker-compose` with `vLLM`.