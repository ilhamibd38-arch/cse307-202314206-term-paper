"""Learned eviction policy.

Idea: for each resident page we compute simple features from the PAST only
(recency, frequency, ...). A small classifier predicts "is this page a good
victim?". The training label comes from Belady's rule: a page is a good victim
if its next use is the farthest in the future (labels use the future, features
never do -- no leakage). At eviction time we evict the page with the highest
predicted probability.
"""
import bisect
from collections import Counter, deque
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

INF = 10 ** 9
FEATURES = ["recency", "freq_window", "prev_gap"]   # stationary features only


class NextUse:
    """Where is page p referenced next, at or after time t? (Used only for labels / logging.)"""

    def __init__(self, trace):
        self.pos = {}
        for i, p in enumerate(trace):
            self.pos.setdefault(p, []).append(i)

    def after(self, p, t):
        lst = self.pos.get(p, [])
        k = bisect.bisect_left(lst, t)
        return lst[k] if k < len(lst) else INF


class PageStats:
    """Past-only statistics per page."""

    def __init__(self, window=50):
        self.W = window
        self.last, self.prev_gap, self.count, self.loaded = {}, {}, {}, {}
        self.win, self.wcount = deque(), Counter()

    def access(self, p, t):
        self.prev_gap[p] = t - self.last[p] if p in self.last else 1000
        self.last[p] = t
        self.count[p] = self.count.get(p, 0) + 1
        self.win.append(p)
        self.wcount[p] += 1
        if len(self.win) > self.W:
            self.wcount[self.win.popleft()] -= 1

    def features(self, q, t):
        return [t - self.last[q],          # recency: refs since last access
                self.wcount[q],            # frequency in the last W references
                self.prev_gap[q]]          # gap between its last two accesses


def collect_training_data(trace, frames, window=50):
    """Replay the trace (LRU decides evictions so states are realistic) and, at every
    eviction, record features of each resident page + Belady label."""
    nu, stats, mem = NextUse(trace), PageStats(window), set()
    X, y = [], []
    for t, p in enumerate(trace):
        if p not in mem:
            if len(mem) == frames:
                cands = list(mem)
                nxt = [nu.after(q, t) for q in cands]
                far = max(nxt)
                for q, n in zip(cands, nxt):
                    X.append(stats.features(q, t))
                    y.append(int(n == far))
                mem.remove(min(cands, key=lambda q: stats.last[q]))
            mem.add(p)
            stats.loaded[p] = t
        stats.access(p, t)
    return X, y


def train_model(traces, frames_list, kind="tree", window=50):
    X, y = [], []
    for tr in traces:
        for f in frames_list:
            a, b = collect_training_data(tr, f, window)
            X += a
            y += b
    if kind == "tree":
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(np.array(X), np.array(y))
    return model


def learned_policy(trace, frames, model, window=50, log=None):
    """Run the learned policy. If `log` is a list, every eviction decision is appended to it
    (the future is used ONLY to annotate the log for evaluation, never for the decision)."""
    stats, mem, hits = PageStats(window), set(), []
    nu = NextUse(trace) if log is not None else None
    for t, p in enumerate(trace):
        if p in mem:
            hits.append(True)
        else:
            hits.append(False)
            if len(mem) == frames:
                cands = list(mem)
                X = np.array([stats.features(q, t) for q in cands])
                proba = model.predict_proba(X)[:, 1]
                score = proba + 1e-9 * X[:, 0]          # tie-break: older access wins
                k = int(np.argmax(score))
                victim = cands[k]
                if log is not None:
                    nxts = [nu.after(q, t) for q in cands]
                    log.append(dict(t=t, page_in=p, victim=victim,
                                    conf=float(proba[k]),
                                    victim_feat=[float(v) for v in X[k]],
                                    cand_recency=[float(v) for v in X[:, 0]],
                                    victim_next=nxts[k], max_next=max(nxts),
                                    median_next=float(np.median(nxts))))
                mem.remove(victim)
            mem.add(p)
            stats.loaded[p] = t
        stats.access(p, t)
    return np.array(hits)
