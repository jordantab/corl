# data-science-team7-s2024

## Dataset

Code samples ([`dataset/`](dataset/)) and scores for initial fine-tuning sourced from the [PIE dataset](https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset). You can download the dataset directly [here](https://drive.google.com/file/d/19IL3VETwVI9rdibB979Xm4gEWYwn0CkV/view).

The `improvement_pairs_additional_metadata.csv` dataset's columns are in the following format:

```
user_id, problem_id, language, submission_id_v0, submission_id_v1, cpu_time_v0, cpu_time_v1, memory_v0, memory_v1, status_v0, status_v1, improvement_frac, code_v0, code_v1
```

A processed version can be
A version processed for initial fine tuning (with the optimization instruction) can be found [here](https://drive.google.com/drive/folders/1GILiI_7I6tt-QmI6jhQkfgcCTHsmYh9O?usp=sharing).

Or you can process the CSV file yourself using [`process_csv.py`](process_csv.py):

```bash
python3 process_csv.py --infile datasets/improvement_pairs_additional_metadata.csv --outfile datasets/improvement_pairs_instruction.json
```

Other uses of the script are suggested in the script's file header.

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
