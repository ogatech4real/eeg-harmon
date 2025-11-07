# Re-export legacy API so old imports keep working
from .neuroCombat import (  # noqa: F401
    neuroCombat,
    neuroCombatFromTraining,
    make_design_matrix,
    adjust_data_final,
)
