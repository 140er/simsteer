"""Camera intrinsics + extrinsics for projecting model-space points back
onto the captured frame.

Coordinate conventions (matching openpilot's "device frame"):
    X = forward (m)
    Y = lateral, +left (m)
    Z = vertical, +up (m)

A road-plane point at (X, Y, 0) projects to pixel (u, v) via a pinhole
camera mounted at (0, 0, height_m) above the road, looking along +X with
pitch about +Y (positive pitch tilts the camera down).

    u = -fx * Y / X + cx
    v =  fy * (height - Z) / X + cy        (with pitch corrections folded in)

This is correct only when the road is flat and the camera is centered
laterally. Both are reasonable for ETS2 highway cruising.

The calibration is persisted to `calibration.json` at the project root so
it survives restarts. Tune it live in the overlay with the keys defined
in main.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pilot.paths import load_with_fallback, state_path

# Back-compat aliases: existing call sites and tools (tools/test_per_game_state.py)
# import these names from pilot.calibration. The canonical implementations live
# in pilot.paths now.
resolve_state_path = state_path
_load_with_fallback = load_with_fallback

CONFIG_PATH = state_path("calibration")


@dataclass
class Calibration:
    """Camera setup for the game we're driving.

    The model-input pipeline ([pilot/preprocess.py], [pilot/warp.py])
    warps each captured frame onto a virtual camera with fixed
    intrinsics (openpilot's medmodel / sbigmodel) using a homography
    parameterized by `fov_h_deg`, `pitch_deg`, and `yaw_deg`. Height
    doesn't enter the warp (the model's virtual camera is canonical)
    but it sets where `road_to_image` places the road plane in FRD,
    so it directly affects the overlay's vertical alignment.

    Sign conventions (matching openpilot's `rpyCalib`):
      pitch_deg > 0  =>  camera nose pitched DOWN
      yaw_deg   > 0  =>  camera nose pitched LEFT
    Roll is forced to 0 just like openpilot does.
    """
    image_w: int = 3440
    image_h: int = 1440
    # HFOV of the captured frame (your in-game FOV setting). Drives
    # the captured frame's K matrix in `pilot.warp.camera_intrinsics`.
    fov_h_deg: float = 90.0
    # Camera height above road, m. road_to_image uses this to place
    # the road plane at z_FRD = +height_m; LiveCalib refines it from
    # `road_transform[2]` once calStatus = CALIBRATED.
    height_m: float = 1.0
    pitch_deg: float = 0.0    # positive = camera looks DOWN
    yaw_deg: float = 0.0      # positive = camera looks LEFT
    # Flip-on-output sign. The model's plan uses openpilot's left-positive
    # convention; ETS2/AC/Forza all map "positive axis" to "stick right",
    # so a model-frame left turn (positive curvature) must be flipped to
    # send "stick left" to the pad. -1.0 is the correct default for every
    # game we currently support — leaving it at +1 makes both the overlay
    # (lanes draw mirrored) and the steering output flipped on fresh
    # installs. Toggle via M hotkey or Camera-tab checkbox if your game
    # uses the opposite convention.
    lateral_sign: float = -1.0
    # Principal-point location as a fraction of (image_w, image_h). Game
    # captures put the optical center dead-center (0.5, 0.5). The model's
    # virtual camera (medmodel) puts the horizon at cy=47.6 in a 256-tall
    # frame -> cy_frac = 47.6 / 256 = 0.186. `model_view_calib` overrides
    # these so the overlay's horizon math is consistent with what the
    # model was trained to see.
    cx_frac: float = 0.5
    cy_frac: float = 0.5
    # ---- crop (now WIRED — applied by warp_to_model) ----
    # Fraction of the source frame to drop from the top / bottom
    # before warping. Use for interior cab cameras where the cab roof
    # / hood obscures sky / road and the warp ends up sampling that
    # dark content into the model's "above horizon" or "below horizon"
    # regions. Sliders on the Camera tab; bounded to [0, 0.45] each.
    crop_top_pct: float = 0.0
    crop_bottom_pct: float = 0.0
    # ---- warp mode ----
    # True (default): use the simple crop+resize path — no perspective
    # transform, just take the central HFOV slice the model expects,
    # crop to 2:1 aspect, resize. Cheaper and reads correctly for
    # screen-captured game frames where there's no real lens to
    # rectify. False: use the full openpilot-style perspective warp
    # with pitch/yaw rotation. Toggle via the Camera tab.
    simple_warp: bool = True
    # ---- legacy / unused (kept so old JSON loads don't error) ----
    # `model_*_fov_h_deg` were used by an older pre-warp pipeline; the
    # current warp reprojects to fixed medmodel / sbigmodel intrinsics
    # so these are ignored. `synthesize_wider_fov` was never implemented
    # — the warp doesn't synthesize anything. Use `crop_top_pct` /
    # `crop_bottom_pct` to fit interior-cab sources to the model's
    # expected vertical content distribution instead.
    model_narrow_fov_h_deg: float = 52.0
    model_wide_fov_h_deg: float = 120.0
    synthesize_wider_fov: bool = False

    # ---- rpyCalib accessor for the model-input warp ----

    def rpy_calib_rad(self) -> tuple[float, float, float]:
        """`[roll, pitch, yaw]` in radians — the `device_from_calib`
        Euler that `pilot.warp.get_warp_matrix` consumes."""
        return (0.0,
                math.radians(self.pitch_deg),
                math.radians(self.yaw_deg))

    # ---- derived ----

    @property
    def fov_v_deg(self) -> float:
        # Square pixels: tan(fov_v/2) = tan(fov_h/2) * H/W.
        return math.degrees(2 * math.atan(
            math.tan(math.radians(self.fov_h_deg) / 2) * self.image_h / self.image_w))

    @fov_v_deg.setter
    def fov_v_deg(self, value: float) -> None:
        """Set FOV by its vertical angle. Most games (AC, Forza, GTA,
        BeamNG, Project Cars) expose VFOV in their settings — this
        setter lets the tuner show the slider in the units the user
        already has in front of them.

        Internally we still store `fov_h_deg` (camera matrices are
        easier to think about that way), but the conversion uses the
        current capture aspect: HFOV = 2·atan(tan(VFOV/2)·W/H)."""
        v = max(1.0, min(179.0, float(value)))
        # Aspect comes from the most-recent captured frame (image_w /
        # image_h are updated by update_for_frame each tick).
        aspect = self.image_w / max(self.image_h, 1)
        h_rad = 2 * math.atan(math.tan(math.radians(v) / 2) * aspect)
        self.fov_h_deg = math.degrees(h_rad)

    @property
    def fx(self) -> float:
        return (self.image_w / 2) / math.tan(math.radians(self.fov_h_deg) / 2)

    @property
    def fy(self) -> float:
        return (self.image_h / 2) / math.tan(math.radians(self.fov_v_deg) / 2)

    @property
    def cx(self) -> float:
        return self.image_w * self.cx_frac

    @property
    def cy(self) -> float:
        return self.image_h * self.cy_frac

    # ---- projection ----

    def road_to_image(self, xyz: np.ndarray) -> np.ndarray:
        """Project Nx3 road-frame points to Nx2 pixel coords.

        Points behind the camera get NaN — caller should filter them
        before drawing.

        Implements openpilot's single-matrix projection chain (see
        `selfdrive/ui/onroad/augmented_road_view.py`):

            M = K · view_from_device · device_from_calib(rpy)
            pt_FRD = (X,  Y · lateral_sign,  height_m − Z)
            pixel_homog = M @ pt_FRD
            u, v = pixel_homog[:2] / pixel_homog[2]

        The point is first converted to the FRD device frame: the
        camera sits at the origin (so the road plane lives at
        z_FRD = +height_m); `lateral_sign` flips the model's native
        left-positive Y to FRD right-positive; the Z input is +up
        relative to the road (Z=0 on flat road), so `height_m − Z`
        gives the FRD-down coordinate.

        The previous formulation decomposed pitch and yaw into two
        separate 2D rotations and missed the `sin(yaw)·sin(pitch)·h`
        cross-terms — alignment was visibly off whenever both
        mount angles were non-zero (which is always after LiveCalib
        converges).
        """
        # Local import keeps `pilot.warp` independent at module load —
        # warp imports Calibration for typing only.
        from pilot.warp import VIEW_FROM_DEVICE, rot_from_euler

        xyz = np.atleast_2d(xyz).astype(np.float64)
        pt_frd = np.column_stack([
            xyz[:, 0],
            xyz[:, 1] * self.lateral_sign,
            self.height_m - xyz[:, 2],
        ])

        K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        device_from_calib = rot_from_euler(self.rpy_calib_rad())
        M = K @ VIEW_FROM_DEVICE @ device_from_calib  # 3x3

        homog = pt_frd @ M.T  # (N, 3)
        depth = homog[:, 2]
        valid = depth > 0.5
        u = np.where(valid, homog[:, 0] / depth, np.nan)
        v = np.where(valid, homog[:, 1] / depth, np.nan)
        return np.stack([u, v], axis=1)

    # ---- persistence ----

    @classmethod
    def load(cls, game: str | None = None,
             path: Path | None = None) -> "Calibration":
        """Load this game's calibration. Falls back to the legacy
        single-file `calibration.json` if a per-game file doesn't
        exist yet (smooth migration). Pass `path` to override the
        resolution entirely (used in tests)."""
        if path is None:
            path = load_with_fallback("calibration", game)
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items()
                          if k in cls.__dataclass_fields__})
        return cls()

    def save(self, game: str | None = None,
             path: Path | None = None) -> None:
        if path is None:
            path = state_path("calibration", game)
        path.write_text(json.dumps(asdict(self), indent=2))

    def update_for_frame(self, frame_shape: tuple[int, int]) -> None:
        """Sync image_w/image_h to the frame we're actually capturing."""
        h, w = frame_shape[:2]
        self.image_w = w
        self.image_h = h


def polyline_to_image(calib: Calibration, x_idxs: np.ndarray, y_offsets: np.ndarray,
                      z_offsets: np.ndarray | None = None) -> np.ndarray:
    """Convenience: take parallel arrays (X, Y[, Z]) and return Nx2 pixel coords."""
    if z_offsets is None:
        z_offsets = np.zeros_like(y_offsets)
    xyz = np.stack([x_idxs, y_offsets, z_offsets], axis=1)
    return calib.road_to_image(xyz)


def model_view_calib(captured_calib: Calibration,
                     captured_shape: tuple[int, int],
                     cropped_shape: tuple[int, int],
                     view_w: int, view_h: int,
                     model_height_m: float | None = None) -> Calibration:
    """Calibration for projecting model predictions onto the model's own
    warped input view.

    The model input is warped to openpilot's medmodel virtual camera
    (fl=910, cx=256, cy=47.6 in 512x256). After the warp, mount pitch
    and yaw are corrected by construction — the model's output sits in
    the calibrated frame. Drawing those outputs back onto the view
    therefore uses medmodel intrinsics with rpy=0.

    `model_height_m`: the camera height the MODEL believes it has
    (typically `decoded.road_transform[2]`, ~1.2 m for comma-3-trained
    weights even when our actual mount is taller). The horizontal
    projection (u) doesn't depend on height, but the vertical (v) does
    — using the captured calib's `height_m` here causes lane lines to
    draw at the wrong row when our truck-cab mount is much taller than
    what the model assumes. Pass the model's per-frame estimate so the
    overlay's lanes land on the visible road markings.
    """
    # Local import keeps `pilot.warp` independent of `pilot.calibration`
    # at module load — both files reference each other for typing.
    from pilot.warp import (MEDMODEL_CX, MEDMODEL_CY, MEDMODEL_FL,
                            MEDMODEL_INPUT_SIZE)
    medmodel_w, medmodel_h = MEDMODEL_INPUT_SIZE
    # HFOV that medmodel was trained to expect.
    medmodel_fov_h_deg = 2 * math.degrees(math.atan(medmodel_w / (2 * MEDMODEL_FL)))
    if model_height_m is None:
        model_height_m = captured_calib.height_m
    return Calibration(
        image_w=view_w,
        image_h=view_h,
        fov_h_deg=medmodel_fov_h_deg,
        height_m=float(model_height_m),
        pitch_deg=0.0,                  # warp corrected it
        yaw_deg=0.0,                    # warp corrected it
        lateral_sign=captured_calib.lateral_sign,
        cx_frac=MEDMODEL_CX / medmodel_w,
        cy_frac=MEDMODEL_CY / medmodel_h,
    )
