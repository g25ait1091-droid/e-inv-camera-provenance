"""Where the v2 scripts find data, third-party code and outputs.

Set these environment variables, or accept the defaults (relative to this repository):
  EINV_V2       working root with out/ (shipped results), data/ (archived v1 generations), paper/ (figure
                output) and logs/.                                  default: <repo>/v2/workspace
  EINV_MYDRIVE  directory containing forensic_datasets/ (Dresden_Exp etc.; see DATA.md) and, optionally,
                inv_channel/ (the study archive).                   default: <repo>/MyDrive
  EINV_EXT      third-party checkouts: DiffusionShield/, prnu-python/, noiseprint/ (see v2/README.md).
                                                                    default: <repo>/v2/ext
  EINV_DATA     the primary study's generated images: gens/ (adapter seeds 0-2) and gens_ext/ (seeds
                3-11), tens of GB, so point this at a disk with room.
                                                                    default: $EINV_V2/data
  EINV_TMP      scratch for large intermediates that are not results (the C7 residual stack is ~0.5 GB);
                nothing here is read back after a run.              default: $EINV_V2/tmp
"""
import os
SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SRC, "..", ".."))
V2 = os.environ.get("EINV_V2", os.path.join(REPO, "v2", "workspace"))
MYDRIVE = os.environ.get("EINV_MYDRIVE", os.path.join(REPO, "MyDrive"))
DATASETS = os.path.join(MYDRIVE, "forensic_datasets")
EXT = os.environ.get("EINV_EXT", os.path.join(REPO, "v2", "ext"))
DATA = os.environ.get("EINV_DATA", os.path.join(V2, "data"))
GENS = os.path.join(DATA, "gens")
GENS_EXT = os.path.join(DATA, "gens_ext")
TMP = os.environ.get("EINV_TMP", os.path.join(V2, "tmp"))
for _d in ("out", "paper", "logs"):
    os.makedirs(os.path.join(V2, _d), exist_ok=True)
