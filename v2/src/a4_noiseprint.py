"""
A4 / G1b slot 3 -- noiseprint wrapper.

noiseprint (Cozzolino & Verdoliva, TIFS 2020; grip-unina/noiseprint) with the
PUBLISHED per-QF weights. Used as a camera-MODEL fingerprint extractor; the
slot-3 detector is a nearest-reference classifier over noiseprint correlations.

WHY THIS SATISFIES THE DESIGN PRINCIPLE
    detector statistic : correlation of a LEARNED CNN noiseprint against
                         per-model noiseprint references
    retention statistic: classical PRNU PCE (Mihcak wavelet residual) against a
                         per-device fingerprint from held-out naturals
Different extractor, different reference class, different statistic. Slot 3 is
therefore not circular in the way G1a's PRNU slot was.

PERFORMANCE-ONLY DEVIATION, verified
    noiseprint.genNoiseprint opens a tf.Session and calls saver.restore on EVERY
    call, which would dominate a sweep. This module restores once per QF net and
    reuses the session. The graph, weights and feed are the repo's own; only the
    session lifetime differs. verify_against_reference() checks the outputs are
    bit-identical to genNoiseprint on sample images.

COMPATIBILITY SHIMS, both non-algorithmic
    `tensorflow` is aliased to tensorflow.compat.v1 (the repo does
    `import tensorflow as tf` against TF 1.2).
    PIL.JpegImagePlugin.convert_dict_qtables is restored as an identity; modern
    Pillow returns plain lists, which is what the function had become.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
import sys

import numpy as np

REPO = os.path.join(EINV.EXT, 'noiseprint')
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import tensorflow.compat.v1 as tf                       # noqa: E402
tf.disable_eager_execution()
sys.modules["tensorflow"] = tf

import PIL.JpegImagePlugin as _jpg                      # noqa: E402
if not hasattr(_jpg, "convert_dict_qtables"):
    _jpg.convert_dict_qtables = lambda qtables: qtables

from noiseprint import noiseprint as _np_mod            # noqa: E402
from noiseprint.utility.utilityRead import imread2f, jpeg_qtableinv  # noqa: E402

_SESSIONS = {}


def qf_of(path):
    """JPEG quality factor as the repo derives it; 101 when unavailable."""
    try:
        q = jpeg_qtableinv(path)
    except Exception:
        return 101
    q = int(round(float(q)))
    return 101 if q > 100 else max(51, min(101, q))


def _session(qf):
    """One restored session per QF net, reused across calls."""
    if qf not in _SESSIONS:
        ckpt = _np_mod.chkpt_folder % ("net", qf)
        sess = tf.Session(config=_np_mod.configSess)
        _np_mod.saver.restore(sess, ckpt)
        _SESSIONS[qf] = sess
    return _SESSIONS[qf]


def extract(img_f32, qf):
    """
    Noiseprint of a single-channel float image in [0,1].

    Mirrors genNoiseprint's small-image branch: one forward pass of net.output
    with the image as [1,H,W,1]. Inputs are kept under the repo's own
    `largeLimit` so the tiled branch is never entered.
    """
    assert img_f32.ndim == 2, img_f32.shape
    assert img_f32.size <= _np_mod.largeLimit, (
        "image %s exceeds noiseprint largeLimit %d; the tiled branch would apply"
        % (img_f32.shape, _np_mod.largeLimit))
    sess = _session(qf)
    out = sess.run(_np_mod.net.output,
                   feed_dict={_np_mod.x_data: img_f32[np.newaxis, :, :, np.newaxis]})
    return np.squeeze(out).astype(np.float32)


def verify_against_reference(paths, crop_fn, n=3):
    """
    R6-style check that reusing the session changes nothing.

    Compares extract() against the repo's own genNoiseprint on the same inputs.
    """
    from noiseprint.noiseprint import genNoiseprint
    worst = 0.0
    checked = 0
    for p in paths[:n]:
        img, _mode = imread2f(p, channel=1)
        img = crop_fn(img)
        qf = qf_of(p)
        a = extract(img, qf)
        b = np.squeeze(np.asarray(genNoiseprint(img, qf), dtype=np.float32))
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        worst = max(worst, d)
        checked += 1
    return worst, checked


def ncc(a, b):
    """Normalised correlation at zero shift, both mean-removed."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return 0.0 if nx == 0 or ny == 0 else float(np.dot(x, y) / (nx * ny))


def ncc_fft(a, b):
    """Second route for R6: same quantity via FFT, sharing no code with ncc."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    cc = np.fft.irfft2(np.fft.rfft2(x) * np.conj(np.fft.rfft2(y)), s=x.shape)
    nx, ny = np.linalg.norm(x.ravel()), np.linalg.norm(y.ravel())
    return 0.0 if nx == 0 or ny == 0 else float(cc[0, 0] / (nx * ny))
