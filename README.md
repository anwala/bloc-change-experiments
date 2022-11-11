# BLOC Change Experiments
### Prerequisites

* Install BLOC from `cluster` branch
  ```bash
  $ git clone -b cluster https://github.com/anwala/bloc.git
  $ cd bloc/; pip install .; cd ..; rm -rf bloc;
  ```

### Task 1: Run change analysis on groups of accounts from command line

```
$ bloc change -o change_report.jsonl --token-pattern=word -m 8 --no-sort-action-words --bloc-alphabets action --bearer-token="$BEARER_TOKEN" OSoMe_IU acnwala
```

### Task 2: Run change analysis on groups of tweets

```bash
$ python bloc_change.py --bc-keep-bloc-segments --add-pauses --bloc-model=word --tweets-path=../datasets --task evaluate rtbust
```

To run analysis, modify `bloc_change.py::run_tasks()`