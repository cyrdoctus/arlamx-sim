# ARLAMX v2.7.5

ARLAMX is a spacecraft plant written in C++ and a trainer written in Python. The plant integrates orbit and attitude under free-molecular aerodynamics, solar and Earth radiation pressure, a gravity field, and magnetic control. The trainer wraps that plant in a Gymnasium environment and trains a small neural attitude advisor with Stable-Baselines3.

It does not use Basilisk. A local Basilisk install can act as an optional gravity referee in `tests/basilisk_ref/`.

The short guide is a dark-mode site. Open [`docs/guide/index.html`](docs/guide/index.html) in a browser. It covers the commands, a free-molecular run for a spacecraft, the onboard control loop and deep-RL training, STL simplification, and the Python libraries. Equation-level notes live in [`docs/html/index.html`](docs/html/index.html). The full flag list is [`docs/notes/commands.md`](docs/notes/commands.md).

## What it is for

Three ways to use the same plant:

1. **Free-molecular flow.** Turn a mesh into flat plates and integrate drag, lift, and radiation pressure along an orbit. Shipped runs use the hex sail or the SolarCat plate model. `python main.py decay` is the front door.
2. **Onboard control.** An outer advisor picks a target attitude every few minutes. An inner law (MRP or quaternion) tracks it with magnetorquers or torque rods at a 2 s step. Scripted advisors (min drag, heuristic, sampling MPC) use the same loop as a trained network.
3. **Deep RL.** PPO, SAC, or TD3 learn that outer advisor. Networks stay small (default 4 layers × 16 units) so an INT8 copy can be aimed at a microcontroller. Training is FP32. Quantization is a separate command.

## Install and build

Python 3.12. The intended install is the conda environment in `environment.yml`. That file pins a CPU PyTorch build. Do not `pip install -r requirements.txt` into that environment: an unpinned `torch` wheel replaces the conda build and breaks Stable-Baselines3.

```bash
mamba env create -n arlamx -f environment.yml   # or: mamba env update -n arlamx -f environment.yml
mamba activate arlamx
./build.sh
python main.py test
```

`./build.sh` compiles `cpp/` into `python/arlamx_v2/arlamx_cpp*.so`. It refuses anything other than Python 3.12. Point it at a specific interpreter with `ARLAMX_PYTHON`.

Coefficient files (GGM03S, World Magnetic Model) and the shipped meshes are resolved from `data/`, or from `ARLAMX_GGM`, `ARLAMX_WMM`, `ARLAMX_HEX_GEOM`, and `ARLAMX_SOLARCAT_STL` when those are set.

## Commands

```bash
python main.py list
python main.py test
python main.py verify

python main.py decay                                      # hex sail, config/orbit.yaml
python main.py decay --kind min --set orbit.simulation.geom=stl1pct

python main.py sim geometry --in craft.stl --quality 1pct # mesh -> plates
python main.py sim lift_drag_study --out outputs/results/lift_drag

python main.py train --network config/network_quick.yaml  # short PPO smoke run
python main.py train --physics high --gsi cll --controller quaternion
python main.py quantize --in outputs/models/<id>/models/ppo_<id>.zip --out int8.zip
```

Settings live in `config/network.yaml`, `config/orbit.yaml`, `config/train.yaml`, and `config/plant/`. Every training session writes a snapshot under `outputs/snapshots/` that can be replayed with `--from-snapshot`.

## Layout

```
main.py           front door
cpp/              plant: aero, radiation, gravity, attitude, magnetorquers
python/arlamx_v2/ Gym env, geometry simplifier, trainers
config/           network, orbit, train, and plant YAML
data/             GGM03S, WMM, hex .geom, SolarCat mesh
docs/guide/       user guide (open index.html)
docs/html/        physics reference
docs/notes/       development writing (plans, changelogs, handoffs)
tests/            one folder per module
```

`outputs/` is created when you run. It is gitignored, along with `build/` and the compiled extension.

## Libraries

Direct Python packages: NumPy, SciPy, PyYAML, Matplotlib, Gymnasium, Stable-Baselines3, PyTorch, pymsis, trimesh, cascadio, pybind11, TensorBoard, cloudpickle, pytest, CMake. The plant itself is C++17. The table and the reason for each package are on the [libraries page](docs/guide/libraries.html).

## License

[MIT](LICENSE). You may use, modify, and redistribute the code if you keep the copyright notice and the permission notice, which names the original copyright holder. Extra attribution, including the Basilisk citation for the spherical-harmonic gravity segments, is in [NOTICE](NOTICE). Development notes are in [`docs/notes/`](docs/notes/README.md). Assistants used during development are listed in [AI_USE.md](AI_USE.md).

`data/GGM03S.txt` and `data/WMM*.COF` are third-party coefficient files and are not covered by the MIT license.

## Publishing

This tree is set up to push as-is.

- `.gitignore` leaves out `build/`, `__pycache__/`, `outputs/` (campaign runs are about 800 MB), and the compiled `arlamx_cpp` extension. Rebuild with `./build.sh` after a clone.
- **GitHub Pages:** Settings → Pages → Deploy from a branch → folder `/docs`. The site root is [`docs/index.html`](docs/index.html).
- **GitLab Pages:** `.gitlab-ci.yml` publishes `docs/` from the default branch.
- **CI:** `.github/workflows/ci.yml` and the GitLab `check` job confirm the guide pages exist. A full plant build and `python main.py test` are manual (`workflow_dispatch` on GitHub, a manual job on GitLab) because they install the conda environment.

## Version

v2.7.5 evaluates GGM03S through degree 70 with fully normalised Legendre functions. Training presets stay at degree 2, 4, and 8. History: [`docs/notes/CHANGELOG_v2.7.md`](docs/notes/CHANGELOG_v2.7.md), [`docs/notes/CHANGELOG_v2.6.md`](docs/notes/CHANGELOG_v2.6.md), [`docs/notes/CHANGELOG_v2.5.md`](docs/notes/CHANGELOG_v2.5.md), [`docs/notes/CHANGELOG_v2.1.md`](docs/notes/CHANGELOG_v2.1.md). The spherical-harmonic evaluator is checked against Basilisk from the Autonomous Vehicle Systems Laboratory at the University of Colorado Boulder. Cite that model when you reuse `cpp/src/orbit/gravity.cpp`. The plant does not link Basilisk.
