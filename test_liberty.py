from ethics_synthesis_agent import _normalize_mfq_scales, _is_frontend_libertarian_profile

assert _normalize_mfq_scales({"care_harm":0.0})["care_harm"] == 1.0
assert _normalize_mfq_scales({"care_harm":6.5})["care_harm"] == 6.0
assert _is_frontend_libertarian_profile({"norm_profile":"Libertarians (US)"}) is True
assert _is_frontend_libertarian_profile({"norm_profile":"Liberals (US)"}) is False