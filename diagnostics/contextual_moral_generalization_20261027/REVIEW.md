# First experiment interpretation

The shared feature-weight policy transferred to two previously unseen cue
combinations. Both learned arms selected disclosure for the coercive protective
emergency and protection for the noncoercive harmful emergency across all five
seeds. The fixed coercion heuristic selected the wrong response by the simulator's
own outcome model in both contexts. Expected functional success was 0.950 for
the learned arms and 0.035 for this deliberately limited control.

This is a mechanism demonstration with a weak baseline and deliberately
adversarial test contexts for that baseline. The experiment supplies informative
purpose annotations and encodes the value of protection/rescue in its reward
function. It therefore does not establish new moral reasoning, independent
justification of the rewards, reliable interpretation of natural-language cases,
or superiority to a fixed Parliament. Four wording variants are the same feature
vector to the policy, so they do not establish language robustness. Five seeds
measure training randomness, not five new ethical problem families.

Validation was less successful: both learned arms made the toy-optimal choice
70% of the time, versus 50% for the fixed heuristic. The remaining errors include
unnecessary investigation under an established protective purpose and failure
to investigate unresolved purpose. Expected success was 0.7485 rather than the
validation optimum of 0.825. These failures remain in the CSV and SQLite journal;
neither the cue schema nor hyperparameters were tuned after seeing them.

Retrieval did not improve the reported validation or test choices over shared
weights alone. It did retrieve observed successful and unsuccessful training
examples, with isolated memory and explicit blend contributions. That makes
it inspectable, but does not establish a benefit. Preserve the simpler
weight-only control in the next comparison.

The independent process proxy scores the fixed heuristic 1.0 on the test contexts
despite its poor outcomes, because those particular contexts have established
purpose and therefore do not require investigation by the defined proxy. This
shows why that narrow procedural measure must not be interpreted as general
ethical quality or combined into an unexamined moral reward.

The next bounded experiment should use a small shadow set of already admitted
Parliament cases, with human-audited action/cue mappings and separately specified
consequences. Compare against an actual frozen Parliament selection only where
such a selection exists; do not turn provisional preferences into decisions.
First test whether those input annotations and independent feedback can be
trusted. Keep learned rankings advisory until that bridge is demonstrated.
Automatic agent recruitment, live decision changes, new action discovery and
large new case domains would confound this first transfer test.
