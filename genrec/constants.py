"""Fixed SPRINT settings (not configurable)."""

# Semantic IDs: N_DIGIT codes per item, each from a CODEBOOK_SIZE codebook,
# built by OPQ on sentence embeddings PCA-reduced to SENT_EMB_PCA dims.
N_DIGIT = 4
CODEBOOK_SIZE = 256
SENT_EMB_PCA = 256

# History window fed to the encoder (most recent items).
MAX_ITEM_SEQ_LEN = 20
# Minimum history length for a training window.
MIN_HIST_LEN = 1

# Size of the catalog top-k pulled before truncating to the metric's top-k.
N_CANDIDATES = 20
