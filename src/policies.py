"""Classical page replacement policies: FIFO, LRU, Optimal (Belady).

Every function takes (trace, frames) and returns a numpy bool array,
hits[i] == True if reference i was a hit. Keeping the per-reference result lets
us split the statistics before / after the workload shift.
"""
from collections import OrderedDict, deque
import numpy as np

INF = 10 ** 9


def fifo(trace, frames):
    queue, resident, hits = deque(), set(), []
    for p in trace:
        if p in resident:
            hits.append(True)
        else:
            hits.append(False)
            if len(queue) == frames:
                resident.remove(queue.popleft())   # evict the oldest loaded page
            queue.append(p)
            resident.add(p)
    return np.array(hits)


def lru(trace, frames):
    mem, hits = OrderedDict(), []                  # order: least -> most recently used
    for p in trace:
        if p in mem:
            hits.append(True)
            mem.move_to_end(p)
        else:
            hits.append(False)
            if len(mem) == frames:
                mem.popitem(last=False)            # evict least recently used
            mem[p] = True
    return np.array(hits)


def optimal(trace, frames):
    n = len(trace)
    nxt, last = [INF] * n, {}
    for i in range(n - 1, -1, -1):                 # nxt[i] = next index using trace[i]
        nxt[i] = last.get(trace[i], INF)
        last[trace[i]] = i
    resident, hits = {}, []                        # page -> index of its next use
    for i, p in enumerate(trace):
        if p in resident:
            hits.append(True)
        else:
            hits.append(False)
            if len(resident) == frames:
                victim = max(resident, key=resident.get)   # used farthest in future
                del resident[victim]
        resident[p] = nxt[i]
    return np.array(hits)
