# BigBenchHard --- Zero-Shot Chat-Template No-Options

This is a reversion of the original task, were questions were slightly modifies to be zero-shot, no guidance with system prompts and without options (where possible).

There are 25 tasks here, as in the base dataset [PNYX/bbh](https://huggingface.co/datasets/PNYX/bbh).

## Original Paper
Title: `Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them`
Abstract: https://arxiv.org/abs/2210.09261

A suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH).
These are the task for which prior language model evaluations did not outperform
the average human-rater.

Homepage: https://github.com/suzgunmirac/BIG-Bench-Hard


## Citation
```
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
```

### Groups, Tags, and Tasks

#### Groups

- `bbh_split`: Is the a-vert based zero-shot chat-completions version.


#### Tags

- `bbh_split-all`

#### Tasks

- `bbh-split_01-boolean_expressions`
- `bbh-split_02-causal_judgement`
- `bbh-split_03-date_understanding`
- `bbh-split_04-disambiguation_qa`
- `bbh-split_05-dyck_languages`
- `bbh-split_06-formal_fallacies`
- `bbh-split_07-geometric_shapes`
- `bbh-split_08-hyperbaton`
- `bbh-split_09-logical_deduction_five_objects`
- `bbh-split_10-logical_deduction_seven_objects`
- `bbh-split_11-logical_deduction_three_objects`
- `bbh-split_12-movie_recommendation`
- `bbh-split_13-multistep_arithmetic_two`
- `bbh-split_14-navigate`
- `bbh-split_15-object_counting`
- `bbh-split_16-penguins_in_a_table`
- `bbh-split_17-reasoning_about_colored_objects`
- `bbh-split_18-ruin_names`
- `bbh-split_19-salient_translation_error_detection`
- `bbh-split_20-snarks`
- `bbh-split_21-sports_understanding`
- `bbh-split_22-temporal_sequences`
- `bbh-split_23-tracking_shuffled_objects_five_objects`
- `bbh-split_24-tracking_shuffled_objects_seven_objects`
- `bbh-split_25-tracking_shuffled_objects_three_objects`

### Checklist

- [x] Is in Eval-harness v1.0 ?
- [ ] Has been checked for regression from v1.0?
- [ ] Has been checked for equivalence with original paper methodology?
- [ ] "Main" checked variant clearly denoted?
