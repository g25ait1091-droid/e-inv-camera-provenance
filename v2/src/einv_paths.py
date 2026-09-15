"""Where the v2 scripts find data, third-party code and outputs.

Set these environment variables, or accept the defaults (relative to this repository):
  EINV_V2       working root with out/ (shipped results), data/ (archived v1 generations), paper/ (figure
                output) and logs/.                                  default: <repo>/v2/workspace
  EINV_MYDRIVE  directory containing forensic_datasets/ (Dresden_Exp etc.; see DATA.md) and, optionally,
                inv_channel/ (the study archive).                   default: <repo>/MyDrive
  EINV_EXT      third-party checkouts: DiffusionShield/, prnu-python/, noiseprint/ (see v2/README.md).
                                                                    default: <repo>/v2/ext
"""
import os
SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
V2 = os.environ.get("EINV_V2", os.path.join(REPO, "v2", "workspace"))
MYDRIVE = os.environ.get("EINV_MYDRIVE", os.path.join(REPO, "MyDrive"))
DATASETS = os.path.join(MYDRIVE, "forensic_datasets")
EXT = os.environ.get("EINV_EXT", os.path.join(REPO, "v2", "ext"))
for _d in ("out", "paper", "logs"):
    os.makedirs(os.path.join(V2, _d), exist_ok=True)
