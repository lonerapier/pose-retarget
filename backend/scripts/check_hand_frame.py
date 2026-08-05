"""Checks mapper.hand_orientation() — the SEW-Mimic stage-3 input `H`.

Run:  just hand-check

Hand orientation is the easiest thing here to get subtly, invisibly wrong: a sign
error still produces a perfectly well-formed rotation matrix, it just describes
the wrong hand, and no amount of staring at a webcam overlay will show it. So
it's checked against properties rather than eyeballs.

  1. Round-trip. Build landmarks that *should* yield a given rotation and confirm
     hand_orientation() recovers it, over random rotations, both hands. Subsumes
     orthonormality, right-handedness and equivariance.

  2. Mirror symmetry. A left hand posed as the exact mirror of a right hand must
     give the mirrored rotation, M·R·diag(1,-1,1) for the sagittal reflection M.
     This catches the left/right sign flip, the most likely bug in it.

  3. Degeneracy. A hand whose knuckle line has collapsed onto the finger
     direction has no observable roll, and must return None rather than a
     confident answer built out of noise.

  4. The axis permutation round-trips, so from_robot_axes() here can't silently
     disagree with mapper.to_robot_axes().
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retarget.mapper import (  # noqa: E402
    HandLandmarks,
    Landmark,
    hand_orientation,
    to_robot_axes,
)

N_SAMPLES = 300
TOL = 1e-9

# Sagittal reflection in robot axes: +y points to the subject's left.
MIRROR = np.diag([1.0, -1.0, 1.0])

HAND_LEN = 0.05
KNUCKLE_HALF = 0.022


def from_robot_axes(v: np.ndarray) -> np.ndarray:
    """Inverse of to_robot_axes, which is (-dz, dx, -dy)."""
    return np.array([v[1], -v[2], -v[0]])


def lm(v: np.ndarray) -> Landmark:
    return Landmark(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def make_hand(R: np.ndarray, side: str, wrist=np.array([0.5, 0.4, 0.0])):
    """Landmarks for a hand whose orientation should come back as exactly R.

    Inverts hand_orientation's convention: z is -f (fingers), and y is the
    knuckle line k on the right hand, -k on the left.
    """
    f = -R[:, 2]
    k = (1.0 if side == "right" else -1.0) * R[:, 1]

    def place(offset: np.ndarray) -> Landmark:
        return lm(wrist + from_robot_axes(offset))

    return HandLandmarks(
        wrist=lm(wrist),
        middle_mcp=place(HAND_LEN * f),
        index_mcp=place(0.85 * HAND_LEN * f + KNUCKLE_HALF * k),
        pinky_mcp=place(0.85 * HAND_LEN * f - KNUCKLE_HALF * k),
        thumb_tip=place(0.5 * HAND_LEN * f + 1.5 * KNUCKLE_HALF * k),
        index_tip=place(2.0 * HAND_LEN * f + KNUCKLE_HALF * k),
    )


def report(label: str, worst: float, tol: float = TOL) -> bool:
    ok = worst < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<44} worst {worst:.3e}")
    return ok


def main() -> int:
    rng = np.random.default_rng(0)
    rotations = Rotation.random(N_SAMPLES, random_state=0).as_matrix()
    results = []

    # --- 4: the permutation this file assumes matches the one in mapper ---
    worst = max(
        float(np.abs(to_robot_axes(from_robot_axes(v)) - v).max())
        for v in rng.normal(size=(50, 3))
    )
    results.append(report("axis permutation round-trips", worst))

    # --- 1: round-trip, both hands ---
    for side in ("left", "right"):
        worst = 0.0
        for R in rotations:
            recovered = hand_orientation(make_hand(R, side), side)
            assert recovered is not None
            worst = max(worst, float(np.abs(recovered - R).max()))
        results.append(report(f"{side} hand: R -> landmarks -> R", worst))

    # --- 2: mirror symmetry ---
    worst = 0.0
    for R in rotations:
        right = make_hand(R, "right")

        # Reflect the right hand's landmarks across the sagittal plane and read them
        # as a left hand. Reflecting in robot axes and mapping back is the same as
        # reflecting in landmark space, and much easier to be sure of.
        def reflect(p: Landmark) -> Landmark:
            return lm(from_robot_axes(MIRROR @ to_robot_axes(p.as_array())))

        left = HandLandmarks(
            **{
                name: reflect(getattr(right, name))
                for name in HandLandmarks.model_fields
            }
        )
        got = hand_orientation(left, "left")
        assert got is not None
        worst = max(worst, float(np.abs(got - MIRROR @ R @ MIRROR).max()))
    results.append(report("mirrored right hand == mirrored rotation", worst))

    # --- 3: degenerate inputs return None ---
    # Built by hand: make_hand() can't produce a degenerate case, which is the
    # point of it.
    origin = np.array([0.5, 0.4, 0.0])
    fingers = np.array([1.0, 0.0, 0.0])

    def along(scale: float) -> Landmark:
        return lm(origin + from_robot_axes(scale * HAND_LEN * fingers))

    f_only = HandLandmarks(
        wrist=lm(origin),
        middle_mcp=along(1.0),
        # Index and pinky knuckles at the same point: no knuckle line at all.
        index_mcp=along(0.85),
        pinky_mcp=along(0.85),
        thumb_tip=lm(origin),
        index_tip=lm(origin),
    )
    collapsed = HandLandmarks(
        **{name: lm(origin) for name in HandLandmarks.model_fields}
    )
    # Knuckle line parallel to the fingers: well-separated landmarks, but the
    # roll about the hand is genuinely unobservable.
    parallel = f_only.model_copy(
        update={"index_mcp": along(0.9), "pinky_mcp": along(0.7)}
    )
    degenerate = {
        "no knuckle line": f_only,
        "all landmarks coincident": collapsed,
        "knuckle line parallel to fingers": parallel,
    }
    for label, hand in degenerate.items():
        for side in ("left", "right"):
            got = hand_orientation(hand, side)
            ok = got is None
            results.append(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  degenerate ({side}): {label}")

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
