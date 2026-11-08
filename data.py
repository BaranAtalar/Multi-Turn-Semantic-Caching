"""Workloads: conversation states, their embeddings and costs, and the judge's labels.

A *state* is one user request together with the conversation before it. Every state has

* a request q (the user's message) and a history H (the earlier turns),
* the reference answer a, the response a cache entry made from this state would hold,
* a generation cost: the input and output tokens of the LLM call, normalized to [0, 1].

The data of each workload lives in data/<workload>/ (see download.py):

    states.parquet               the states, their splits and token counts
    embeddings.npz               Granite embeddings of the request, history, answer and recent context
    judge_labels.parquet         the LLM judge's verdict for every (request, cached state) pair used
    risk_training_pairs.parquet  the calibration pairs the reuse-risk model is trained on
"""
import html
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WORKLOADS = {"taskmaster3": "Taskmaster-3", "socrateach": "SocraTeach"}

SEED = 7
LAMBDA_REQUEST, LAMBDA_HISTORY = 0.7, 0.3   # weights of the request and history halves of the state x
CONTEXT_MESSAGES, CONTEXT_CHARS = 6, 200     # ElastiCache-MT's context string: last 6 messages, 200 characters each
COST_QUANTILES = (0.01, 0.99)                # cost normalization: 1st and 99th percentiles of the calibration costs


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def context_string(history_pairs, query):
    """ElastiCache-MT's cache key: the recent messages, each truncated, with role prefixes."""
    messages = []
    for user_text, assistant_text in history_pairs:
        messages += [("user", user_text), ("assistant", assistant_text)]
    messages.append(("user", query))
    return " | ".join(f"{role}: {clean_text(text)[:CONTEXT_CHARS]}" for role, text in messages[-CONTEXT_MESSAGES:])


class Workload:
    """One workload. States keep the order of states.parquet; arrays are indexed by that position."""

    def __init__(self, name):
        self.name, self.label = name, WORKLOADS[name]
        folder = DATA / name
        if not folder.exists():
            raise FileNotFoundError(f"{folder} is missing: run `python download.py` first.")
        df = pd.read_parquet(folder / "states.parquet").reset_index(drop=True)
        for col in ["query", "reference_answer", "history_text"]:
            df[col] = df[col].fillna("").astype(str)
        self.df = df
        self.n = len(df)
        self.state_ids = df["state_id"].astype(str).to_numpy()
        self.position = {s: i for i, s in enumerate(self.state_ids)}
        self.conv_ids = df["conversation_id"].astype(str).to_numpy()
        self.conv_code = pd.factorize(df["conversation_id"])[0]

        split = df["split"].astype(str).str.lower().str.strip().to_numpy()
        self.cal = np.flatnonzero(split == "calibration")
        self.val = np.flatnonzero(split == "validation")
        self.test = np.flatnonzero(split == "test")
        self.src = np.sort(np.concatenate([self.cal, self.val]))   # the source pool for test-time caches
        self.split = split

        # Turn order within each conversation
        turn = df["turn_idx"].astype(int).to_numpy()
        self.sorted_order = np.lexsort((turn, df["usable_conv_idx"].astype(int).to_numpy()))
        self.rank_of = np.empty(self.n, dtype=np.int64)
        self.rank_of[self.sorted_order] = np.arange(self.n)
        pos = df.iloc[self.sorted_order].groupby("conversation_id", sort=False).cumcount().to_numpy()
        self.pos_in_conv = pos[self.rank_of]

        self._text_keys()
        self._embeddings(folder)
        self._costs()
        self.labels = JudgeLabels(folder / "judge_labels.parquet", self.position)
        self.risk_pairs = pd.read_parquet(folder / "risk_training_pairs.parquet")

    def _text_keys(self):
        queries, answers = self.df["query"].to_numpy(), self.df["reference_answer"].to_numpy()
        context = np.empty(self.n, dtype=object)
        history, previous = [], None
        for g in self.sorted_order:
            if self.conv_ids[g] != previous:
                history, previous = [], self.conv_ids[g]
            context[g] = context_string(history, queries[g])
            history.append((queries[g], answers[g]))
        q_clean = np.array([clean_text(x) for x in queries], dtype=object)
        a_clean = np.array([clean_text(x) for x in answers], dtype=object)
        self.context_code = pd.factorize(pd.Series(context, dtype=object))[0]       # identical recent context
        self.query_code = pd.factorize(pd.Series(q_clean, dtype=object))[0]         # identical request
        self.query_lower_code = pd.factorize(pd.Series([q.lower() for q in q_clean], dtype=object))[0]
        self.qa_code = pd.factorize(pd.Series([q + "\x00" + a for q, a in zip(q_clean, a_clean)]))[0]

    def _embeddings(self, folder):
        z = np.load(folder / "embeddings.npz", allow_pickle=True)
        if not np.array_equal(z["state_ids"].astype(str), self.state_ids):
            raise RuntimeError("embeddings.npz is not aligned with states.parquet")
        self.E_query = l2_normalize(z["query"])        # the request
        self.E_history = l2_normalize(z["history"])    # the earlier turns
        self.E_answer = l2_normalize(z["answer"])      # the reference answer
        self.E_context = l2_normalize(z["context"])    # ElastiCache-MT's context string
        # The state x = [sqrt(0.7) e(q), sqrt(0.3) e(H)], normalized
        self.X = l2_normalize(np.concatenate(
            [np.sqrt(LAMBDA_REQUEST) * self.E_query, np.sqrt(LAMBDA_HISTORY) * self.E_history], axis=1))

    def _costs(self):
        df = self.df
        h, q, a = (df[c].to_numpy(dtype=float) for c in ["history_tokens", "query_tokens", "answer_tokens"])
        self.history_tokens = h
        raw = h + q + a
        lo = float(np.quantile(raw[self.cal], COST_QUANTILES[0]))
        hi = float(np.quantile(raw[self.cal], COST_QUANTILES[1]))
        span = max(hi - lo, 1e-12)
        self.cost = np.clip((raw - lo) / span, 0.0, 1.0)   # c(z): normalized cost of calling the LLM
        self.cost_in = (h + q - lo) / span                 # the part known before the call (input tokens)
        self.cost_out = self.cost - self.cost_in           # the part that is learned (output tokens)


class JudgeLabels:
    """The judge's verdict for (request, cached state) pairs: 1 = acceptable reuse, 0 = unacceptable."""

    def __init__(self, path, position):
        df = pd.read_parquet(path)
        t = df["target_state_id"].map(position).to_numpy()
        c = df["candidate_state_id"].map(position).to_numpy()
        self.n_states = len(position)
        self.accept = dict(zip(t.astype(np.int64) * self.n_states + c.astype(np.int64),
                               df["acceptable"].astype(bool).to_numpy()))

    def lookup(self, t, c):
        """1.0 / 0.0 for judged pairs, NaN for pairs without a label."""
        keys = np.asarray(t, dtype=np.int64) * self.n_states + np.asarray(c, dtype=np.int64)
        get = self.accept.get
        return np.array([np.nan if (v := get(k)) is None else float(v) for k in keys.tolist()])


@lru_cache(maxsize=None)
def load(name):
    return Workload(name)
