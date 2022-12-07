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
T | ⚃T | ⚄T | ⚂T⚂T⚁T⚁T⚁T⚂T | ⚃T | ⚂T⚂T | ⚃T⚂T | ⚂T | ⚃T | ⚃T⚂T | ⚃T | ⚃T | ⚃T | ⚃T | ⚃T | 
⚂T | ⚂T | ⚄T | ⚃T | ⚃T | ⚂T | ⚃T | ⚃T⚁T | ⚃T | ⚃T | ⚃T⚂T | ⚂T | ⚄T | ⚂T | ⚃T | ⚃T⚂T | ⚄T 
| ⚃T | ⚃T⚁T | ⚂T | ⚄T⚂T | ⚃T | ⚃p | ⚃T | ⚃T | ⚃T | ⚃T | ⚂T□T□T⚁T | ⚃T | ⚄T | ⚄T⚂T⚁π | ⚃T 
| ⚃T | ⚄T | ⚃T | ⚃T | ⚄T | ⚄T | ⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T
```
Upon inspection, we observe that `@jesus`' actions mostly involve tweeting (`T`) and pausing (e.g., for weeks`⚃`). With the exception of the last week (`⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T`), it rarely engaged in conversations or replies to tweets (`p`). More importantly, the BLOC string for the last week looks very different the others. In other words, the last week marks a change (maybe temporary) in behavior which coincides with when the account gained became verified.

##### Case study: @storygraphbot

[@storygraphbot](https://twitter.com/storygraphbot) is a Twitter bot (created by Alexander Nwala in October 2018) that runs every hour, tracking top news stories and creating tweet threads (collections of tweets) that report updates (rising/falling/same attention) of the stories. Before the account was controlled by software, the (human) creator used to manually posted tweets. On May 2, 2021, this changed, and the account became controlled by software.

Consider the BLOC action strings for `@storygraphbot` generated from tweets posted by the account between November 6, 2018 and May 10, 2021.
```
⚄Tπππππππππππ | Tππππ⚂T⚁p⚀π⚁p | ⚄T | ⚃r | ⚄T | ⚄r | ⚄T | ⚂r | ⚄T⚁p | ⚂r⚂r | ⚃r | ⚃rT | 
⚄r⚀T | ⚄p⚂T | ⚄T⚂π | ⚃r⚂r | ⚂Tπππ⚁r | ⚃r | ⚃r | ⚄p | ⚃r | ⚄r□p | ⚂r | ⚄rr | ⚄TππππππTπππππ
ππππππππ□TππππππππππππππππππππππππππππππππππππππππππππππππππππππππππTππππππππ⚁Tπππππππππππ
ππππππππππππππππππππππTTπππππππππππ⚁T | ⚁π⚀π⚁π⚀π⚁π⚁π⚁π⚁Tππ⚀π⚀π⚁π⚁π⚀π⚁Tππππ⚁π⚀π⚁T⚀π⚁π⚁T⚀π⚁π
⚁T⚀π⚁π⚀π⚁π⚁π⚁T⚀π⚀π⚁π⚀π⚁π⚁π⚀Tπ⚀π⚁π⚂T⚁π⚀π⚁π⚂T⚁π⚀π⚁π⚁π | ⚁T⚁π⚁Tπππππππππ
```
Upon further inspection of `@storygraphbot`'s BLOC string, we can observe strings generated when the human controlled the account,
```
⚄Tπππππππππππ | Tππππ⚂T⚁p⚀π⚁p | ⚄T | ⚃r | ⚄T | ⚄r | ⚄T | ⚂r | ⚄T⚁p | ⚂r⚂r | ⚃r | ⚃rT | 
⚄r⚀T | ⚄p⚂T | ⚄T⚂π | ⚃r⚂r | ⚂Tπππ⚁r | ⚃r | ⚃r | ⚄p | ⚃r | ⚄r□p | ⚂r | ⚄rr 
```
are different from those generated when the account was controlled by a bot:
```
⚄TππππππTπππππππππππππ□TππππππππππππππππππππππππππππππππππππππππππππππππππππππππππTπππππππ
π⚁TπππππππππππππππππππππππππππππππππTTπππππππππππ⚁T | ⚁π⚀π⚁π⚀π⚁π⚁π⚁π⚁Tππ⚀π⚀π⚁π⚁π⚀π⚁Tππππ⚁π
⚀π⚁T⚀π⚁π⚁T⚀π⚁π⚁T⚀π⚁π⚀π⚁π⚁π⚁T⚀π⚀π⚁π⚀π⚁π⚁π⚀Tπ⚀π⚁π⚂T⚁π⚀π⚁π⚂T⚁π⚀π⚁π⚁π | ⚁T⚁π⚁Tπππππππππ
```

**The main purpose of our change detection task is detect BLOC strings that are quite different from their neighbors. For example, `⚄T` is quite different from `⚁p□p⚂T⚂p⚀T⚀p⚀p⚀p⚀p⚁p⚀p⚁p□p⚀p□p⚁T⚁T⚀p⚀p□T⚁T⚁T`.**

The motivation for detecting change is: since we expect that the behaviors of typical social media users is homogeneous, users that exhibit heterogeneous behaviors are suspicious. There quantifying change in behavior could be useful in identifying social media manipulation similar to bot detection and coordination detection.

### BLOC change command

The BLOC `change` command was implemented to aid in identifying adjacent BLOC strings (e.g., `⚄rr` vs `⚄TππππππTπππππππππππππ□T`) that are "significantly" different. We define "significant" later. But for now, consider the following steps that summarize the method of measuring the similarity between adjacent (neighbors) BLOC strings.

1. BLOC strings (e.g., `⚄r□p | ⚂r | ⚄rr | ⚄rr | ⚄TππππππTπππππππππππππ`) are segmented into weekly bins; week 1: - `⚄r□p`, week 2: `⚂r`, week 3: `⚄rr`, week 4: `⚄rr`, and week 5: `⚄TππππππTπππππππππππππ`. Then we compute similarity between neighbors (e.g., `⚄r□p` vs. `⚂r` or `⚂r` vs. `⚄rr`).

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
| `⚄rr`      |     `⚄rr`  |         1.00      |
| `⚄rr`      |     `⚄TππππππTπππππππππππππ`   |         0.12      |

### How much change is significant?

From the previous section, the four similarity values (0.21, 0.00, 1.00, and 0.12) indicate four degrees of change. For example, the BLOC string from week 3 and week 4 are the same (no change) resulting in a cosine similarity value of 1.00. In contrast, the BLOC strings from weeks 2 and week 3 have no character in common (100% change) resulting in a cosine similarity value of 0.00. The question remains, what cosine similarity value indicates significant change? If we chose 0.00, this means we would only flag neighbors that have no characters in common, but this might seems too strict. 

Our first solution at identifying significant change starts with the following theory: `The change in BLOC strings follows a normal distribution.` We conjecture that change mostly hovers around a central value (the mean) and might deviate slightly away from the mean. Our theory is inspired by the rationale that the activities of typical social media users is homogeneous. In other words typical social media users have repetitive behaviors. Therefore, we could use our knowledge of the normal distribution (the mean µ and standard deviation σ) of the cosine similarity of adjacent BLOC strings for a sample of account s (e.g., 1000 human accounts) to identify change values that are significant. Since 68% of values are within one standard deviation of a normal distribution, change values that are outside this range could be considered significant. In other words, z-scores of cosine similarity value that exceed 1 could be considered significant.

### Task 1: Run change analysis on account from command line

Also try this:
```bash
$ bloc change --timeline-startdate="2022-11-13 23:59:59" --change-mean=0.7 --change-stddev=0.3 --token-pattern=word -m 8 --no-sort-action-words --bloc-alphabets action --bearer-token="$BEARER_TOKEN" jesus
```

### Assignment 1: Run change analysis on groups of tweets

```bash
$ python bloc_change.py --bc-keep-bloc-segments --add-pauses --bloc-model=word --tweets-path=../datasets --task evaluate rtbust
```

To run analysis, modify [`bloc_change.py::run_tasks()`](https://github.com/anwala/bloc-change-experiments/blob/a22bd453563db0fa37b755f31b4e4aaeee0a87f0/scripts/bloc_change.py#L167)

* What are the mean, median, and standard deviations of the cosine similarity values for `human` and `bot` rtbust accounts?

### Assignment 2:

Plot a histogram for the distribution of cosine similarity values for `human` and `bot` rtbust accounts