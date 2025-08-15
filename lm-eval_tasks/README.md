# Sample Tasks for `lm-eval`

This folder contains tasks in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) format that implement `a-vert` as a metric.

Please make sure to have deployed a [Text-Embedding-Inference ](https://github.com/huggingface/text-embeddings-inference) endpoint with an embedding model (we recommend [XXX]()), and that the API endpoint is provided in an environment variable like:
```sh
export TEI_ENDPOINT=http://127.0.0.1:8080/embed
```