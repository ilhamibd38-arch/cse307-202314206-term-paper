# CSE-307 Term Paper - Track 1: Learned Page Replacement

**Name:** Tahsin Ilham Raisul   
**ID:** 202314206   
**Course:** CSE-307 (Part B)

## What this project does
1. Implements FIFO, LRU and Optimal (Belady's) page replacement.
2. Adds a lightweight learned eviction policy using past-only recency, recent-window frequency and reuse-gap features.
3. Generates one 5,000-reference synthetic trace with a locality-heavy first phase and random/bursty second phase.
4. Reports hit ratio and page faults before/after the workload shift for 4, 8 and 16 frames.
5. Evaluates explanation confidence and calibration against Belady's choices.

## Setup (Ubuntu)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run
```bash
python3 check_algorithms.py
python3 run_experiments.py
python3 bonus.py
```

The sanity check returned:
`FIFO = 15, LRU = 12, Optimal = 9`.

## Key results
At 8 frames:

| Policy | Before shift | After shift | Hit-ratio drop |
|---|---:|---:|---:|
| FIFO | 0.834 | 0.349 | 0.486 |
| LRU | 0.867 | 0.350 | 0.517 |
| Learned (static) | 0.896 | 0.344 | 0.552 |
| Learned (both phases) | 0.714 | 0.318 | 0.396 |
| Optimal | 0.914 | 0.561 | 0.353 |

The main finding is that locality makes recency useful, but after the shift to mostly random/bursty access, recent use becomes a much weaker predictor of the next reference. The static learned model therefore loses the most of its phase-1 advantage. Training on both phases reduces the drop but does not make the learned policy outperform LRU after the shift.

The bonus evaluation recorded 1,894 eviction decisions. Overall accuracy was 0.188. Mean confidence was 0.484 on correct decisions and 0.479 on incorrect decisions, showing that the confidence score is poorly calibrated. After the shift, accuracy fell to 0.160 while confidence remained about 0.48.

## Experimental design
- Test trace seed: 42.
- Training seeds: 1, 2 and 3, so the test trace is not used for training.
- Frames: 4, 8 and 16.
- Static model: trained only on phase-1 locality data.
- Both-phases model: trained on both workload phases.
- Features are past-only; Belady's future knowledge is used only to create training/evaluation labels.

## AI assistance disclosure
An AI assistant was used to help inspect, help write the code scaffolding, explain the setup steps and run the provided code, organize the experimental results. The experimental results and analysis in the report are my own.

## Results files
The `results/` directory contains the generated CSV files and figures, including the before/after hit-ratio chart, rolling hit-ratio chart, trace figure and calibration figure.

