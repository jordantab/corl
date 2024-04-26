# data-science-team7-s2024

## Setup

You can manage your python environment using a virtual environment:

```bash
python3 -m venv .venv
```

You can activate the environment and install the requirements like so:

```bash
source .venv/bin/activate
pip3 install -r requirements.txt
```

Install the precommit hooks used for python formatting:

```bash
pre-commit install
```

To exit the virtual environment, simply run `deactivate`.

## Dataset

Code samples ([`dataset/`](dataset/)) and scores for initial fine-tuning sourced from the [PIE dataset](https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset). You can download the dataset directly [here](https://drive.google.com/file/d/19IL3VETwVI9rdibB979Xm4gEWYwn0CkV/view).

Public test cases ([`dataset/`](dataset/)) from the [PIE dataset](https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset). You can download the dataset directly [here](https://drive.google.com/file/d/1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO/view?usp=share_link).

Hidden test cases ([`dataset/`](dataset/)) from the [PIE dataset](https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset). You can download the dataset directly [here](https://drive.google.com/file/d/1migwX4wpED0gDDxn7gS6q55vWeXIDgId/view?usp=drive_link).
The `improvement_pairs_additional_metadata.csv` dataset's columns are in the following format:

```
user_id, problem_id, language, submission_id_v0, submission_id_v1, cpu_time_v0, cpu_time_v1, memory_v0, memory_v1, status_v0, status_v1, improvement_frac, code_v0, code_v1
```

Subsets of the data processed for initial fine tuning (with the optimization instruction) can be found [here](https://drive.google.com/drive/folders/1GILiI_7I6tt-QmI6jhQkfgcCTHsmYh9O?usp=sharing).

The following scripts are available for data processing and analysis:

-   [`process_csv.py`](process_csv.py) - Converts the csv dataset to a json file for instruction tuning.
-   [`count_tokens.py`](count_tokens.py) - Analyzes the distribution for input/output sizes (in terms of tokens) ofn instruction tuning dataset (json).
-   [`filter_dataset`](filter_dataset.py) - Creates a subset of an instruction tuning dataset with samples where input/outputs are below the provided threshold.
-   [`train_test_split.py`](train_test_split.py) - Splits a JSON dataset into a train and test set, ensuring that all instances of a particular problem_id are either in the training set or the test set.

You can read more about how to use these scripts in their respective file headers, or by running the
script with the `--help` flag.

## Initial Fine Tuning

The initial fine-tuning with the instruction dataset can be performed with `tune.py`:

```bash
python3 tune.py
```

You may also use commandline flags to override the default configruation. For a complete list of options, check out the file header in [`tune.py`](tune.py) or run:

```bash
python3 tune.py -h
```

The model checkpoint is saved to `models/dataset_tuned_checkpoint`.


## Fine Tuning with Reinforcement Learning 

The Reinforcement Learning fine-tuning can be performed with `rl_run.py`, one can modify the necessary parameters within `rl.sh`. 

### Testing the Script

To test the reinforcement learning script, you can follow these steps:

1. Prepare a test dataset in JSON format that contains input code snippets and their corresponding problem IDs. Place the test dataset file in the specified `DATASET_PATH` location.

2. Run the script using the `rl.sh` bash script.

3. Monitor the output of the script to observe the training progress, including the generated optimized code, rewards, and losses for each episode.

4. After the training is completed, the script will save the trained model checkpoint in the `../checkpoints` directory and generate plots of the training metrics in the `../plots` directory.

5. Evaluate the performance of the trained model by examining the generated optimized code, comparing it with the reference code, and analyzing the training metrics plots.

### Parameters

The following parameters can be specified in the `run_rl.sh` bash script:

- `MODEL_NAME`: The name of the pre-trained model to use for initialization (default: "meta-llama/Meta-Llama-3-8B-Instruct").
- `NUM_EPISODES`: The number of training episodes to run (default: 10).
- `DATASET_PATH`: The path to the dataset file in JSON format (required).
- `MAX_LENGTH`: The maximum length of the input code (default: 256).
- `R1`: The reward value for compilation failure (default: -1.0).
- `R2`: The reward value for unit test failure (default: 0.0).
- `R3`: The reward value for unit test success with no improvement (default: 0.5).
- `R4`: The reward value for unit test success with improvement (default: 1.0).

Adjust these parameters according to your requirements before running the script.



## References

[PPOCoder](https://github.com/reddy-lab-code-research/PPOCoder)

```
@article{shojaee2023ppocoder,
  title={Execution-based code generation using deep reinforcement learning},
  author={Shojaee, Parshin and Jain, Aneesh and Tipirneni, Sindhu and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2301.13816},
  year={2023}
}
```

[Pie Perf Dataset](https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset)

```
@misc{shypula2023learning,
      title={Learning Performance-Improving Code Edits},
      author={Alexander Shypula and Aman Madaan and Yimeng Zeng and Uri Alon and Jacob Gardner and Milad Hashemi and Graham Neubig and Parthasarathy Ranganathan and Osbert Bastani and Amir Yazdanbakhsh},
      year={2023},
      eprint={2302.07867},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

[CodeT5+](https://github.com/salesforce/CodeT5/tree/main)

```
@article{wang2023codet5plus,
  title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
  author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
  journal={arXiv preprint},
  year={2023}
}
```
