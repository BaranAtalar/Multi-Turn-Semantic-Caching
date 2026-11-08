# Multi-Turn Semantic Caching for Low-Cost LLM Serving

Code for the experiments in [Multi-Turn Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2611.01077) (arXiv preprint).

A semantic cache answers a request with the stored response of a similar earlier request instead of calling the LLM. In a conversation, whether a cached response is acceptable depends on both conversations' histories, which embedding similarity predicts poorly. The paper learns the *reuse risk*, the probability that a cached response is unacceptable, from judge feedback, and chooses a budgeted set of conversation states to cache by weighing the generation cost a reuse avoids against a price for unacceptable reuse. CUCB-SC-MT learns the cache from logged conversations; CLCB-LS-MT learns it online while serving.

## Setup

```bash
pip install -r requirements.txt
python download.py
```

`download.py` fetches the prepared workloads (about 260 MB) from the repository's [GitHub release](https://github.com/BaranAtalar/Multi-Turn-Semantic-Caching/releases) into `data/`. They contain everything the experiments need, including the judge's labels, so no API key or GPU is required (see [Data](#data)).

## Reproducing the results

```bash
python run.py offline     # Figure 3 (a, c)
python run.py online      # Figure 3 (b, d), Figure 4 and Table 1 (needs the offline results)
python plot.py            # results/figure3.pdf and results/figure4.pdf
```

`python run.py all` runs both parts, and `--workload taskmaster3` or `--workload socrateach` runs one workload. Results go to `results/<workload>/`:

| File | Paper |
|---|---|
| `offline_loss.csv` | Figure 3 (a, c) and Tables 13 and 15: test loss at γ = 1 with 95% CIs, and the validation-selected setting of every method and budget |
| `offline_paired_differences.csv` | The offline differences in Section 6.1: CUCB-SC-MT minus each baseline (pp of the no-cache loss) |
| `online_loss.csv` | Figure 3 (b, d), Tables 14 and 16, and Table 1 (precision, false reuse and safe cost avoided; Table 1 is the `1000` rows) |
| `online_paired_differences.csv` | The online differences in Section 6.1: CLCB-LS-MT minus each baseline |
| `online_losses.npz` | The loss of every request in stream order (Figure 4 is its running ratio to the cost) |

The offline part takes a few hours for Taskmaster-3 and about an hour for SocraTeach on a laptop CPU; most of it is scoring the reuse risk of every pair of ε-net centers. The online part takes about two to three hours for Taskmaster-3 and under an hour for SocraTeach.

## Setting

* **Workloads.** *Taskmaster-3*: 2,200 movie-ticketing dialogues with at least two user turns (21,892 requests); the responses are the logged agent turns. *SocraTeach*: five dialogues for each of 440 math problems (2,200 dialogues, 10,821 requests) between a GPT-4 tutor and a simulated student. Each workload is split by conversation into 1,800 calibration, 200 validation and 200 test conversations.
* **State.** A request q with its history H is embedded with Granite Embedding 311M Multilingual R2 as x = [√0.7 e(q); √0.3 e(H)], normalized. Distances are ‖x − y‖/2 ∈ [0, 1].
* **Cost.** The input and output tokens of the request's LLM call (GPT-4o-mini tokenizer), normalized to [0, 1] by the 1st and 99th percentiles of the calibration costs. The input part is known before the call; the output part is learned.
* **Acceptability.** GPT-4o-mini (temperature 0) judges whether a cached response is acceptable for a request and its history. All methods share the judge and its labels.
* **Loss.** A miss costs c(z); a hit costs γ = 1 if the judge rejects the reuse and 0 otherwise. The loss is reported as a percentage of the no-cache loss Σ c(z), with 95% intervals from a bootstrap over conversations (2,000 resamples). Paired differences use the same resamples.
* **Offline protocol.** Every hyperparameter (ε-net radius, risk multiplier γ̂, similarity thresholds, vCache's δ) minimizes the validation loss of a cache built from the calibration conversations; the selected setting is rebuilt from calibration and validation conversations and evaluated on the test conversations. Settings with more than 1% unjudged validation hits are skipped. Cache budgets are k = 100, 200, 300, 500, 1000 and no budget.
* **Online protocol.** All 2,200 conversations arrive in one random order (seed 108), turns in order, and every method starts from an empty cache. Methods observe the cost of each LLM call and the judge label of each reuse and keep their offline-selected hyperparameters; CLCB-LS-MT uses γ̂ = γ = 1 and the most common offline ε.

## Algorithms

| Method | How it caches and serves |
|---|---|
| CUCB-SC-MT (offline) | Covers the calibration states with an ε-net, weighs each center by its share of held-out states, bounds the center's cost from above (c̄ = c_in + ĉ_out + β_c σ_c) and the risk of serving one center with another's response from above (R̄ = min{1, γ̂ (r̂ + β_r σ_r)}), both by kernel ridge regression, and picks k centers by Reverse Greedy. A request reuses the cached response with the lowest R̄ when it is below its own c̄. |
| CLCB-LS-MT (online) | Grows the ε-net while serving and learns cost and risk online with lower confidence bounds (reuse iff c̲ > min R̲), so uncertain pairs keep being reused and labeled. Each conversation is served with a frozen snapshot; the cache is recomputed only at a conversation's end when a center was added, an arrival stage ended, or the cost or risk information grew by ln 2. It also reuses from a buffer of the current conversation's responses. |
| ElastiCache-MT | Thresholds the cosine similarity of an embedding of the last six messages (200 characters each). |
| ContextCache | Serves an identical recent context directly; otherwise retrieves the 100 most similar cached requests and keeps the one whose last five user messages match best, if both similarities pass their thresholds. |
| vCache | Learns a similarity threshold per cached entry from the judge's verdicts on the requests it explores, keeping its error rate below δ. |
| Joint-state threshold | ElastiCache-MT's rule on CUCB-SC-MT's state x. |
| Single-turn (CUCB-SC-Cont) | CUCB-SC-MT's predecessor on the current request alone, with the known mismatch γ̂ d as its risk. |

Offline, ElastiCache-MT, ContextCache and the joint-state threshold cache the k source states that are most often the nearest neighbour of a source state from another conversation, and vCache builds its cache by replaying the source conversations. Online, they insert every LLM response and evict the least recently used entry; vCache inserts a response when its exploration finds the cached one unacceptable.

## Data

`python download.py` creates, for each workload, `data/<workload>/` with

| File | Contents |
|---|---|
| `states.parquet` | One row per request: conversation, split, turn, the request, its history and reference response, and the token counts of its LLM call |
| `embeddings.npz` | Granite embeddings of the request, the history, the reference response and ElastiCache-MT's context string |
| `judge_labels.parquet` | The judge's verdict (and error type) for 307,363 (Taskmaster-3) and 148,342 (SocraTeach) pairs of a request and a cached state, including every pair a method served in the paper's runs |
| `risk_training_pairs.parquet` | The 7,000 calibration pairs the reuse-risk model is trained on |

The requests and responses come from [Taskmaster-3](https://github.com/google-research-datasets/Taskmaster) (CC BY 4.0) and [SocraTeach](https://github.com/CMLF-git-dev/SocraticLM) (CC BY-NC 4.0), and the prepared files keep these licenses.

## Code layout

| File | Contents |
|---|---|
| `data.py` | Loads a workload: states, splits, embeddings, costs and judge labels |
| `models.py` | Nyström kernel ridge regression, the cost model, the pair features ψ and the reuse-risk model |
| `caching.py` | ε-nets, Reverse Greedy, nearest-neighbour lookups and LFU admission |
| `offline.py` | CUCB-SC-MT, the offline baselines, validation selection and the test loss (Figure 3 a, c) |
| `online.py` | CLCB-LS-MT, the online baselines and the stream (Figure 3 b, d, Figure 4, Table 1) |
| `run.py` | Runs the experiments and writes the result tables |
| `plot.py` | Figures 3 and 4 |
| `download.py` | Downloads the prepared data |

## Notes

* A reuse whose (request, cached state) pair has no judge label counts as unacceptable, and a validation setting with more than 1% unjudged hits is skipped. The label tables hold every pair served in the paper's runs, so this only matters where a run makes a different decision. The `unjudged` column of the result tables counts these hits.
* Some decisions depend on how floating-point ties are broken, which varies with the CPU and the BLAS library. This affects the baselines whose caches are chosen by popularity (LFU) with a budget: duplicate requests make the nearest neighbour a tie, so their caches can differ by a few entries between machines. ContextCache's top-100 retrieval and vCache's replay are affected in the same way.
* This code was checked against the paper's notebooks on a different machine (Linux, OpenBLAS), workload by workload:
  * CUCB-SC-MT: every saved test decision at the largest ε (ε*), all budgets and risk multipliers, and the no-budget validation sweep, were identical.
  * Online: CLCB-LS-MT on the full SocraTeach stream made the same decisions at k = 1000, and its no-budget run gave the paper's 74.4% (5,073 reuses). On the first 300 Taskmaster-3 conversations, CLCB-LS-MT, single-turn, ElastiCache-MT, the joint-state threshold and vCache made identical decisions.
  * Single-turn: identical results offline on both workloads, at every budget and with no budget.
  * Offline baselines: with no budget, ElastiCache-MT, vCache and the joint-state threshold matched the paper exactly on both workloads, and so did ContextCache on SocraTeach. With a budget, the results differed by up to 2.5 pp, and by up to 13 pp for ContextCache on Taskmaster-3, where unjudged validation hits changed the selected thresholds.
* The no-budget results cache every center of the ε-net (CUCB-SC-MT, single-turn, CLCB-LS-MT) or every source state (the other methods), with the same validation grids. With no budget, CUCB-SC-MT can use nets of up to 8,000 centers, since it needs no center-to-center risk matrix; with a budget, the limit is 5,000.
* The judge, the embedding model and the token counts are not run here: their outputs are in the prepared data. The latency estimates (Section 6.4) and the appendix studies are not included.

## Citation

```bibtex
@misc{atalar2026multiturn,
  author        = {Atalar, Baran and Beyene, Selly and Liu, Xutong and Joe-Wong, Carlee},
  title         = {Multi-Turn Semantic Caching for Low-Cost {LLM} Serving},
  year          = {2026},
  eprint        = {2611.01077},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
