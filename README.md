# BLOC Change Experiments
### Prerequisites

* Install BLOC from `cluster` branch
  ```bash
  $ git clone -b cluster https://github.com/anwala/bloc.git
  $ cd bloc/; pip install .; cd ..; rm -rf bloc;
  ```

### Introduction

##### Case study: @jesus
[@jesus](https://twitter.com/jesus) is a satirical Twitter account that was created in October 2006 to post tweets around the holidays like Christmas and Easter. On November 9, 2022, to the surprise of many, the account received the blue [verification checkmark](https://twitter.com/jesus/status/1590405986925543424) amid the Twitter's confusing update to the [verification policy](https://www.businessinsider.com/im-verified-jesus-christ-on-twitter-blue-thanks-elon-musk-2022-11) following Elon Musks purchase of the company.

Consider the BLOC action strings for `@jesus` generated from tweets posted by the account between December 8, 2009 and November 11, 2022. Recall that strings generated within the same week are separated by vertical bars `|`:
```
T | ⚃T | ⚄T | ⚂T⚂T⚁T⚁T⚁T⚂T | ⚃T | ⚂T⚂T | ⚃T⚂T | ⚂T | ⚃T | ⚃T⚂T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚂T | ⚃T | ⚃T | ⚃T | ⚃T⚁T | ⚂T⚂T | ⚂T⚂T⚁T | ⚄T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T⚂T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚄T | ⚃T | ⚂T | ⚃T⚁T⚁T⚁T | ⚁T⚁T⚂T | ⚂T⚂T | ⚄T | ⚄T⚂T⚂T | ⚃T | ⚃T | ⚄T | ⚃T | ⚃T | ⚄T | ⚄T | ⚂T⚂T⚁T⚁T⚂T | ⚂T⚂T⚁T⚁T | ⚂T | ⚃T | ⚄T | ⚂T⚁T⚂T | ⚃T⚂T⚁T | ⚃T⚁T | ⚄T | ⚃T | ⚄T⚂T⚂T | ⚄T | ⚂T | ⚂T⚁T | ⚂T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T⚂T⚁T⚂T | ⚃T | ⚄T⚂T | ⚂T | ⚄T | ⚂T | ⚃T | ⚃T⚁T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | ⚂T | ⚃T | ⚂T | ⚃T | ⚄T⚂T | ⚄T | ⚄T | ⚄T | ⚄T | ⚄T | ⚃T⚂T | ⚄T | ⚄T | ⚄T | ⚄T | ⚃T⚂T | ⚂T⚁T⚂T | ⚃T | ⚃T | ⚃T | ⚄T⚁T⚁T⚂T | ⚂T⚂T | ⚃T⚁T | ⚃T | ⚃T | ⚃T | ⚃T | ⚄p | ⚂T⚂T⚁T⚂T | ⚃T | ⚂T⚁T | ⚃T⚀T | ⚃T□T | ⚄T⚁T | ⚃T | ⚃T | ⚂T | ⚃T | ⚄T | ⚃T | ⚃T | ⚃T⚂T | ⚃T | ⚂T | ⚃T⚂T | ⚂T | ⚄T | ⚂T⚁T | ⚂T | ⚄T | ⚃T | ⚁T | ⚄T | ⚃T | ⚃T | ⚃T | ⚂T | ⚃T | ⚄T | ⚂T⚂T | ⚃T⚁T | ⚂T⚂T⚂T | ⚂T | ⚃T | ⚃T | ⚂T | ⚂T⚁T⚂T | ⚂T | ⚂T | ⚄T | ⚃T | ⚃T | ⚂T | ⚃T | ⚃T⚁T | ⚃T | ⚃T | ⚃T⚂T | ⚂T | ⚄T | ⚂T | ⚃T | ⚃T⚂T | ⚄T | ⚃T | ⚃T⚁T | ⚂T | ⚄T⚂T | ⚃T | ⚃p | ⚃T | ⚃T | ⚃T | ⚃T | ⚂T□T□T⚁T | ⚃T | ⚄T | ⚄T⚂T⚁π | ⚃T | ⚃T | ⚄T | ⚃T | ⚃T | ⚄T | ⚄T | ⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T
```
Upon inspection, we observe that `@jesus`' actions mostly involve tweeting (`T`) and pausing (e.g., for weeks`⚃`). With the exception of the last week (`⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T`), it rarely engaged in conversations or replies to tweets (`p`). More importantly, the BLOC string for the last week looks very different the others. In other words, the last week marks a change (maybe temporary) in behavior which coincides with when the account gained became verified.

##### Case study: @storygraphbot

[@storygraphbot](https://twitter.com/storygraphbot) is a Twitter bot (created by Alexander Nwala in October 2018) that runs every hour, tracking top news stories and creating tweet threads (collections of tweets) that report updates (rising/falling/same attention) of the stories. Before the account was controlled by software, the (human) creator used to manually posted tweets. On May 2, 2021, this changed, and the account became controlled by software.

Consider the BLOC action strings for `@storygraphbot` generated from tweets posted by the account between November 6, 2018 and May 10, 2021.
```
⚄Tπππππππππππ | Tππππ⚂T⚁p⚀π⚁p | ⚄T | ⚃r | ⚄T | ⚄r | ⚄T | ⚂r | ⚄T⚁p | ⚂r⚂r | ⚃r | ⚃rT | ⚄r⚀T | ⚄p⚂T | ⚄T⚂π | ⚃r⚂r | ⚂Tπππ⚁r | ⚃r | ⚃r | ⚄p | ⚃r | ⚄r□p | ⚂r | ⚄rr | ⚄TππππππTπππππππππππππ□TππππππππππππππππππππππππππππππππππππππππππππππππππππππππππTππππππππ⚁TπππππππππππππππππππππππππππππππππTTπππππππππππ⚁T | ⚁π⚀π⚁π⚀π⚁π⚁π⚁π⚁Tππ⚀π⚀π⚁π⚁π⚀π⚁Tππππ⚁π⚀π⚁T⚀π⚁π⚁T⚀π⚁π⚁T⚀π⚁π⚀π⚁π⚁π⚁T⚀π⚀π⚁π⚀π⚁π⚁π⚀Tπ⚀π⚁π⚂T⚁π⚀π⚁π⚂T⚁π⚀π⚁π⚁π | ⚁T⚁π⚁Tπππππππππ
```
Upon further inspection of `@storygraphbot`'s BLOC string, we can observe strings generated when the human controlled the account,
```
⚄Tπππππππππππ | Tππππ⚂T⚁p⚀π⚁p | ⚄T | ⚃r | ⚄T | ⚄r | ⚄T | ⚂r | ⚄T⚁p | ⚂r⚂r | ⚃r | ⚃rT | ⚄r⚀T | ⚄p⚂T | ⚄T⚂π | ⚃r⚂r | ⚂Tπππ⚁r | ⚃r | ⚃r | ⚄p | ⚃r | ⚄r□p | ⚂r | ⚄rr 
```
are different from those generated when the account was controlled by a bot:
```
⚄TππππππTπππππππππππππ□TππππππππππππππππππππππππππππππππππππππππππππππππππππππππππTππππππππ⚁TπππππππππππππππππππππππππππππππππTTπππππππππππ⚁T | ⚁π⚀π⚁π⚀π⚁π⚁π⚁π⚁Tππ⚀π⚀π⚁π⚁π⚀π⚁Tππππ⚁π⚀π⚁T⚀π⚁π⚁T⚀π⚁π⚁T⚀π⚁π⚀π⚁π⚁π⚁T⚀π⚀π⚁π⚀π⚁π⚁π⚀Tπ⚀π⚁π⚂T⚁π⚀π⚁π⚂T⚁π⚀π⚁π⚁π | ⚁T⚁π⚁Tπππππππππ
```

**The main purpose of our change detection task is detect BLOC strings that are quite different from their neighbors. For example, `⚄T` is quite different from `⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T`.**

### BLOC change command

The BLOC `change` command was implemented to aid in identifying adjacent BLOC strings (e.g., `⚄rr` vs `⚄TππππππTπππππππππππππ□T`) that are "significantly" different. We define "significant" later. But for now, consider the following steps that summarize the method of measuring the similarity between adjacent (neighbors) BLOC strings.

1. BLOC strings (e.g., `⚄r□p | ⚂r | ⚄rr | ⚄TππππππTπππππππππππππ`) are segmented into weekly bins; week 1: - `⚄r□p`, week 2: `⚂r`, week 3: `⚄rr`, and week 4: `⚄TππππππTπππππππππππππ`. Then we compute similarity between neighbors (e.g., `⚄r□p` vs. `⚂r` or `⚂r` vs. `⚄rr`).

2. Given a pair of neighbor BLOC strings, e.g., `⚄r□p` and `⚂r` from weeks 1 and 2, respectively, to measure their similarity, 
    a. First we convert the strings into TF-IDF vectors:
    ```
    ⚄r□p's vector: [0.53 0.37 0.53 0.00 0.53]
    ⚂r's   vector: [0.00 0.57 0.00 0.81 0.00]
    ```
    b. Second, we compute the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between the vectors, which is `0.21`. A cosine value of `0` means both strings are distinct, a value of `1` means they are exact.

Step 2 is repeated for pairs of neighbors of BLOC strings:
| Neighbor 1 | Neighbor 2 | Cosine similarity |
|:----------:|:----------:|:-----------------:|
| `⚄r□p`     |     `⚂r`   |         0.21      |
| `⚂r`       |     `⚄rr`  |         0.00      |
| `⚄rr`      |     `⚄TππππππTπππππππππππππ`   |         0.12      |

### How much change is significant?

```bash
$ bloc change -o change_report.jsonl --token-pattern=word -m 8 --no-sort-action-words --bloc-alphabets action --bearer-token="$BEARER_TOKEN" OSoMe_IU acnwala
```

Also try this:
```bash
$ bloc change --timeline-startdate="2022-11-13 23:59:59" --change-mean=0.7 --change-stddev=0.3 --token-pattern=word -m 8 --no-sort-action-words --bloc-alphabets action --bearer-token="$BEARER_TOKEN" jesus
```

### Task 2: Run change analysis on groups of tweets

```bash
$ python bloc_change.py --bc-keep-bloc-segments --add-pauses --bloc-model=word --tweets-path=../datasets --task evaluate rtbust
```

To run analysis, modify [`bloc_change.py::run_tasks()`](https://github.com/anwala/bloc-change-experiments/blob/a22bd453563db0fa37b755f31b4e4aaeee0a87f0/scripts/bloc_change.py#L167)