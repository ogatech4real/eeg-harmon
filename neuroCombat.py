# neuroCombat.py  (compat shim)
from __future__ import annotations
from neuroHarmonize import harmonizationLearn, harmonizationApply

def neuroCombat(dat, covars, eb=True, parametric=True, **kwargs):
    """
    Compatibility wrapper that mimics the legacy neuroCombat API.
    Returns (harmonized_data, model) to support callers that expect both.
    """
    model = harmonizationLearn(dat, covars, eb=eb, parametric=parametric)
    harmonized = harmonizationApply(dat, covars, model)
    return harmonized, model
