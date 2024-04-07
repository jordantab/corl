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
python3 process_csv.py --outfile datasets/improvement_pairs_instruction.json
```

Other uses of the script are suggested in the script's file header.

## Initial Fine Tuning

The initial fine-tuning of the dataset can be performed with `tune.py`:

```bash
python3 tune.py
```

The model checkpoint is saved to `models/dataset_tuned_checkpoint`.

## References

-   https://github.com/madaan/pie-perf?tab=readme-ov-file#dataset

```
@article{wang2023codet5plus,
  title={CodeT5+: Open Code Large Language Models for Code Understanding and Generation},
  author={Wang, Yue and Le, Hung and Gotmare, Akhilesh Deepak and Bui, Nghi D.Q. and Li, Junnan and Hoi, Steven C. H.},
  journal={arXiv preprint},
  year={2023}
}
```
