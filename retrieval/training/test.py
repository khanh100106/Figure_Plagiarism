"""
Paper2Fig-2026

Test Early Stopping
"""

from retrieval.training.early_stopping import EarlyStopping

from retrieval.constants import (
    METRIC_RECALL1,
    DIRECTION_MAX,
)

# ============================================================
# Create
# ============================================================

print("=" * 80)
print("Create EarlyStopping")
print("=" * 80)

early = EarlyStopping(

    enabled=True,

    monitor=METRIC_RECALL1,

    direction=DIRECTION_MAX,

    patience=3,

    delta=1e-4,

)

print(early.state_dict())

# ============================================================
# Improvement
# ============================================================

print()
print("=" * 80)
print("Improvement")
print("=" * 80)

scores = [

    0.80,

    0.85,

    0.90,

]

for epoch, score in enumerate(

    scores,

    start=1,

):

    result = early.step(

        epoch=epoch,

        metrics={

            METRIC_RECALL1: score,

        },

    )

    print(

        epoch,

        score,

        result,

    )

# ============================================================
# Patience
# ============================================================

print()
print("=" * 80)
print("Patience")
print("=" * 80)

scores = [

    0.90001,

    0.90002,

    0.90001,

    0.90000,

]

for epoch, score in enumerate(

    scores,

    start=4,

):

    result = early.step(

        epoch=epoch,

        metrics={

            METRIC_RECALL1: score,

        },

    )

    print(

        epoch,

        score,

        result,

    )

    if result["stop"]:

        print()

        print("Early Stop Triggered")

        break

# ============================================================
# State Dict
# ============================================================

print()
print("=" * 80)
print("State Dict")
print("=" * 80)

state = early.state_dict()

print(state)

# ============================================================
# Resume
# ============================================================

print()
print("=" * 80)
print("Resume")
print("=" * 80)

early2 = EarlyStopping(

    enabled=True,

    monitor=METRIC_RECALL1,

    direction=DIRECTION_MAX,

)

early2.load_state_dict(

    state,

)

print(

    early2.state_dict()

)

# ============================================================
# Reset
# ============================================================

print()
print("=" * 80)
print("Reset")
print("=" * 80)

early2.reset()

print(

    early2.state_dict()

)

# ============================================================
# Threshold
# ============================================================

print()
print("=" * 80)
print("Threshold")
print("=" * 80)

early3 = EarlyStopping(

    enabled=True,

    monitor=METRIC_RECALL1,

    direction=DIRECTION_MAX,

    threshold=0.95,

)

scores = [

    0.91,

    0.93,

    0.95,

]

for epoch, score in enumerate(

    scores,

    start=1,

):

    result = early3.step(

        epoch=epoch,

        metrics={

            METRIC_RECALL1: score,

        },

    )

    print(

        epoch,

        score,

        result,

    )

    if result["stop"]:

        print()

        print("Threshold Reached")

        break

print()
print("=" * 80)
print("EarlyStopping Test Finished")
print("=" * 80)