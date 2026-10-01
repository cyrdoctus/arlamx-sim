# tests/

One subfolder per module. Pytest is the runner. Nothing here is written by a training job.

```
tests/
  aero/
  srp/
  orbit/
  attitude/
  control/
  mag/
  geometry/
  integration/
```

Each folder will hold `test_*.py` and tiny `cases/` inputs once implementation starts. Until then this tree is the contract only.

See `../docs/VALIDATION_STANDARD.md` and the matching file in `../docs/modules/`.
