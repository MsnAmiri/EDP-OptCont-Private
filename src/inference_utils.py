DIM_STATE = 26
DIM_ACTION = 6
DIM_LATENT = 8


# ============================================================
# STATE INDICES
# Used by rewards.py
# ============================================================

SPO2_INDEX  = 8 - 1
PAO2_INDEX  = 12 - 1
AWRR_INDEX  = 10 - 1
HR_INDEX    = 5 - 1
IE_INDEX    = 26 - 1
PPLAT_INDEX = 27 - 1
PH_INDEX    = 28 - 1


# ============================================================
# ACTION BOUNDS
# ============================================================

ACTION_BOUNDS = [
    (0, 1),      # FiO2
    (1, 30),     # Pinsp
    (0.1, 3),    # Ti
    (1, 30),     # RR
    (1, 25),     # PEEP
    (0, 1),      # Slope
]


# ============================================================
# TRAINED MODEL CHECKPOINTS
# ============================================================

FP_AUTOENCODER_CKPT = (
    r"C:\Users\bc\Downloads\ventilator2\learning\trained_models"
    r"\autoencoder-2024-05-16-18-50-24.ckpt"
)


FP_LATENT_DYNAMICS_CKPT = (
    r"C:\Users\bc\Downloads\ventilator2\learning\trained_models"
    r"\dynamics-2024-05-16-19-05-13.ckpt"
)