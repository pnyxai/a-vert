# Reasoning-gym 

This is an implementation of the reasoning-gym for using with `lm-eval`.

All the tasks in the base dataset [PNYX/reasoning_gym_lmeh](https://huggingface.co/datasets/PNYX/reasoning_gym_lmeh) are supported:
- gsm_symbolic
- polynomial_equations
- complex_arithmetic
- (!) simple_integration 
- (!) intermediate_integration 
- dice
- (*) codeio

> **(!)** : These tasks are not working correctly with `a-vert`, we are working to improve the matching of math expressions.

> **(*)** : These tasks evaluated using native methods from `reasoning-gym`

The a-vert balanced accuracy against a human annotator, over 100 samples are:
- reasoning_gym_001-gsm_symbolic: 95.85%
- reasoning_gym_002-polynomial_equations: 92.62%
- reasoning_gym_003-complex_arithmetic: 91.67%
- reasoning_gym_004-simple_integration: 60.34%
- reasoning_gym_005-intermediate_integration: 65.59%
- reasoning_gym_006-dice: 97.93%

## Dataset Sources (reasoning-gym)
- **Repository:** https://github.com/open-thought/reasoning-gym
- **Paper:** https://arxiv.org/abs/2505.24760

## Evaluation Mechanics (a-vert)
- **Repository:** https://github.com/pnyxai/a-vert
- **Paper:** https://arxiv.org/abs/2510.01469


### Groups, Tags, and Tasks

#### Groups

- `reasoning_gym-all`: Is the a-vert based zero-shot chat-completions version.

#### Tags

- `reasoning_gym-all`

#### Tasks

- `reasoning_gym_001-gsm_symbolic`
- `reasoning_gym_002-polynomial_equations`
- `reasoning_gym_003-complex_arithmetic`
- `reasoning_gym_004-simple_integration`
- `reasoning_gym_005-intermediate_integration`
- `reasoning_gym_006-dice`
- `reasoning_gym_007-codeio`


### Checklist

- [x] Is in Eval-harness v1.0 ?
- [ ] Has been checked for regression from v1.0?
- [ ] Has been checked for equivalence with original paper methodology?
- [ ] "Main" checked variant clearly denoted?
