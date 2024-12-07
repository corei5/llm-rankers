# MC-NEST

Main paper: https://arxiv.org/pdf/2310.09497 
Dataset: https://huggingface.co/datasets/ustc-zhangzm/HybRank/tree/main

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4-blue)](https://platform.openai.com/)

MC_NEST is a **Monte Carlo Tree Search** framework enhanced with **GPT-4** designed to tackle .........

## Installation

To get started, clone the repository and install the dependencies:

```bash
git clone https://github.com/corei5/MC_NEST_GPT.git
cd MC_NEST_GPT
pip install -r requirements.txt
```

## Usage

Import the Package

```bash
from MC_NEST_GPT.MC_NEST_GPT.MC_NEST_ import MC_NEST_gpt4o
import openai
```

## Set Up OpenAI API Key

Provide your OpenAI API key either directly or through environment variables:

```bash
openai.api_key = 'your_openai_api_key'
```

## Initialize and Run MC_NEST

Define your problem, configure MC_NEST settings, and execute:

```bash
GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
ZERO_SHOT = 1
DUMMY_ANSWER = 2

# Initialize MC_NEST with a problem, number of rollouts, and selection policy
problem = "Let $S$ be a list of positive integers not necessarily distinct in which the number $68$ appears. The average (arithmetic mean) of the numbers in $S$ is $56$. However, if $68$ is removed, the average of the remaining numbers drops to $55$. What is the largest number that can appear in $S$?"
MC_NEST = MC_NEST_gpt4o(problem=problem, max_rollouts=4, selection_policy = IMPORTANCE_SAMPLING, initialize_strategy = ZERO_SHOT)

# Run MC_NEST to get the best answer
best_answer = MC_NEST.run()
print(best_answer)
```
## Customize Selection Policies

Choose from multiple selection policies:

```bash
GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
```


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for major changes.

## License

This project is licensed under the MIT License.

## 📄 Citation

```bash

@article{rabby2024mc,
  title={MC-NEST--Enhancing Mathematical Reasoning in Large Language Models with a Monte Carlo Nash Equilibrium Self-Refine Tree},
  author={Rabby, Gollam and Keya, Farhana and Zamil, Parvez and Auer, S{\"o}ren},
  journal={arXiv preprint arXiv:2411.15645},
  year={2024}
}

```

## References

[1] Zhang, D., Li, J., Huang, X., Zhou, D., Li, Y., & Ouyang, W. (2024). Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B. arXiv preprint arXiv:2406.07394. 



Happy coding! 🚀
