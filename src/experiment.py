"""Shared setup: the ONE test trace used by every policy, plus the trained models."""
from src.trace import generate_trace
from src.learned import train_model

TEST_SEED = 42            # test trace
TRAIN_SEEDS = [1, 2, 3]   # different seeds -> the model never sees the test trace
LENGTH = 5000
FRAMES = [4, 8, 16]


def build_setup(kind="tree"):
    trace, shift = generate_trace(TEST_SEED, LENGTH)
    train = [generate_trace(s, LENGTH) for s in TRAIN_SEEDS]
    # static model: has only ever seen phase-1 (locality) behaviour
    model_static = train_model([tr[:sh] for tr, sh in train], FRAMES, kind)
    # 'both phases' model: trained on both behaviours (what you would get by retraining after the shift)
    model_both = train_model([tr for tr, sh in train], FRAMES, kind)
    return trace, shift, model_static, model_both
