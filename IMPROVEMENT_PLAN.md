# Improvement plan — progress log (WIP, branch `improvements`)

Rule: all model selection uses leave-years-out CV on 2001–2018. The 2019–2025 holdout is scored ONCE at the end.

## Status (updated 2026-10-08)
- [x] Fuels/ERC/Santa Ana weather features (weather.py), ERC reference -> erc_reference.npz
- [x] Ignition points: FPA-FOD 1992-2020 + WFIGS 2021+ (13,773 in county) -> ign / large targets
- [x] Human drivers + WUI + ring features (human.py)
- [x] Panel: 59 features, 3 targets (ign 8,303 / large 164 / fire 1,777 positives)
- [x] Ablation (CV 2001-2018): reports/experiments.csv
- [ ] Optuna tuning (running: tune ign -> fire -> large)
- [ ] Red Flag Warnings download (running)
- [ ] stack / calib / daily experiments -> write models/decisions.json
- [ ] Final single holdout run: src/train.py
- [ ] Forecast run, app check, README, merge + push
- [x] Written: modeling.py, experiments.py, daily.py, redflag.py, monitor.py, new train.py/forecast.py/app.py,
      tests (6 passing incl. forecast/training parity), Dockerfile + render.yaml
