# s4 — one knob at a time from SC_v13 (600 k)

Resume **always** `s3 s42 @ 600192`. Short 400 k. Two seeds. Keep a
checkpoint only if it still clears the promotion bar (B≥0, decay≤16.5,
band≥75, dl≥20, 0 brownouts).

| run | what changes | why |
|---|---|---|
| hold | w_env=0.15 w_clip=0.08 | control — same knobs, more steps |
| clip0.20 | w_clip 0.08→0.20 | punish proposing fights so clip_frac falls |
| env0.25 | w_env 0.15→0.25 | more environment in the training signal |
| env0.35 | w_env 0.15→0.35 | same, one more step |

Do **not** raise decay/dE weights. Do **not** train past the point clip_frac
walks above ~0.5 — that is the s3 collapse. Comparator stays on.

If env0.25 KEeps and decay does not jump, that is the next freeze. If clip0.20
drops clip_frac while holding B and band, mix that w_clip into the next round.

Log: `outputs/v13/logs/s4_train.log`
