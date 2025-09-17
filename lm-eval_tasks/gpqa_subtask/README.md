# GPQA - Subtask

This is a version of GPQA where each domain is evaluated as a separate task.
Also it is implemented for zero-shot chat-completions and a-vert evaluation.
The subtasks are:
- `biology`
- `physics`
- `chemistry`


### Paper

Title: GPQA: A Graduate-Level Google-Proof Q&A Benchmark

Abstract: https://arxiv.org/abs/2311.12022

We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are “Google-proof”). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4–based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions—for example, when developing new scientific knowledge—we need to develop *scalable oversight* methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities.

Homepage: `https://github.com/idavidrein/gpqa/tree/main`

### Citation

```
@misc{rein2023gpqa,
      title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark},
      author={David Rein and Betty Li Hou and Asa Cooper Stickland and Jackson Petty and Richard Yuanzhe Pang and Julien Dirani and Julian Michael and Samuel R. Bowman},
      year={2023},
      eprint={2311.12022},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

This dataset is gated, so you will have to accept the terms of use at https://huggingface.co/datasets/Idavidrein/gpqa and login via `huggingface-cli login` using your HF Hub token before running this task.

### Groups, Tags, and Tasks

#### Groups

None

#### Tags

* `gpqa_subtasks`: runs all GPQA variants.

#### Tasks

* `gpqa_subtask_{main, diamond, extended}_{subtask}`

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
    * [x] Have you referenced the original paper that introduced the task?
