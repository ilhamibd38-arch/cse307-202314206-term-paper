"""Sanity check of the classical algorithms on the standard textbook example."""
from src.policies import fifo, lru, optimal

ref = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
faults = {name: int((~fn(ref, 3)).sum()) for name, fn in
          [("FIFO", fifo), ("LRU", lru), ("Optimal", optimal)]}
print(faults)
assert faults == {"FIFO": 15, "LRU": 12, "Optimal": 9}, "algorithm bug!"
print("OK: matches the textbook values (FIFO 15, LRU 12, Optimal 9)")
