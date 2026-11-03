"""Ownership and durable provenance for one per-game camera/learner pair."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, fields
from functools import wraps
import hashlib
import json
import re
import threading
import time
import uuid

from simsteer import state
from simsteer.paths import state_path

_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def owned(kind="read", *, forwards_deadline=False):
    """Serialize bound methods; detached objects keep their original call forms."""
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            owner = getattr(self, "_geometry_owner", None)
            if owner is None:
                return function(self, *args, **kwargs)
            with owner.lock:
                if owner._internal_read:
                    return function(self, *args, **kwargs)
                return invoke(owner, self, args, kwargs)

        def invoke(owner, self, args, kwargs):
            with owner.operation(deadline=kwargs.get("deadline")):
                if forwards_deadline:
                    kwargs["deadline"] = owner.active_deadline
                if kind in ("load", "rescale", "reset"):
                    owner.revoke(kind)
                elif kind == "helper" and not owner._producer:
                    owner.revoke("private learner mutation")
                if kind == "producer":
                    # An update of another pair cannot authorize this one.
                    if (args[0] if args else kwargs.get("calib")) is not owner.calib:
                        owner.revoke("different calibration pair")
                    before = owner.learner.samples
                    generation = owner.learner.reset_generation
                    owner._producer += 1
                    try:
                        result = function(self, *args, **kwargs)
                    finally:
                        owner._producer -= 1
                    if (owner.learner.reset_generation == generation
                            and owner.learner.samples > before):
                        owner._fresh_samples += owner.learner.samples - before
                    owner.after_update()
                    return result
                result = function(self, *args, **kwargs)
                if kind == "reset":
                    owner.after_reset()
                return result
        return wrapped
    return decorate


def active_deadline(obj, deadline=None):
    owner = getattr(obj, "_geometry_owner", None)
    return owner.active_deadline if owner is not None else deadline


def bind_geometry(calib, learner, game, model, *, persist, flow):
    from simsteer.core.calibration import Calibration
    from simsteer.core.learners.livecalib import LiveCalib
    from simsteer.core.model_identity import pinned_model_identity

    owners = {getattr(obj, "_geometry_owner", None) for obj in (calib, learner)} - {None}
    eligible = (type(calib) is Calibration and type(learner) is LiveCalib
                and persist and type(game) is str and game != "default"
                and re.fullmatch(r"[a-zA-Z0-9_-]+", game) is not None
                and learner._game == game and model == pinned_model_identity())
    if owners:
        owner = next(iter(owners))
        if (not eligible or len(owners) != 1 or owner.calib is not calib
                or owner.learner is not learner or owner.game != game
                or owner.model != model):
            for existing in owners:
                with existing.operation():
                    existing.enabled = False
                    existing.revoke("conflicting pair or model")
            return None
        with owner.operation():
            owner.flow = flow
        return owner
    if not eligible:
        return None
    owner = SharedGeometry(calib, learner, game, model)
    owner.flow = flow
    object.__setattr__(calib, "_geometry_owner", owner)
    learner._geometry_owner = owner
    with owner.operation():
        owner.restore_receipt()
    return owner


class SharedGeometry:
    def __init__(self, calib, learner, game, model, *, epoch_id_factory=None,
                 io_clock=time.monotonic):
        self.calib, self.learner, self.game = calib, learner, game
        self.model = json.loads(json.dumps(model))
        self.flow = None
        self.enabled = True
        self.lock = threading.RLock()
        self._epoch_id_factory = ((lambda: uuid.uuid4().hex)
                                  if epoch_id_factory is None else epoch_id_factory)
        self.io_clock = io_clock
        self.active_deadline = None
        self._depth = self._internal_read = self._producer = 0
        self._publishing = False
        self._issued = set()
        self.epoch = None
        self.trusted = False
        self.reason = "no measured origin"
        self._epoch_input = None
        self._fresh_samples = 0
        self._receipt = None
        self._boundary = self._fingerprint()

    @property
    def paths(self):
        return (state_path("calibration", self.game),
                state_path("livecalib_state", self.game),
                state_path("shared_geometry", self.game))

    def revoke(self, reason):
        """Local only: failure handling must never recurse into persistence."""
        self.trusted = False
        self.epoch = self._epoch_input = self._receipt = None
        self._fresh_samples = 0
        self.reason = str(reason)

    @contextmanager
    def operation(self, *, deadline=None):
        with self.lock:
            outer = self._depth == 0
            if outer:
                self.active_deadline = (state.StateDeadline(self.io_clock() + 2., self.io_clock)
                                        if deadline is None else deadline)
            self._depth += 1
            try:
                if outer:
                    self.audit()
                yield
            except state.StateIOUnavailable as exc:
                self.revoke(exc)
                raise
            except BaseException:
                self.revoke("owned operation failed")
                raise
            finally:
                self._depth -= 1
                if outer:
                    self._boundary = self._fingerprint()
                    self.active_deadline = None

    def _capture(self):
        self._internal_read += 1
        try:
            return asdict(self.calib), self.learner.state_dict()
        finally:
            self._internal_read -= 1

    def _fingerprint(self):
        try:
            return json.dumps(self._capture(), sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, OverflowError, AttributeError):
            return None

    def audit(self):
        fingerprint = self._fingerprint()
        if fingerprint is None or fingerprint != self._boundary:
            self.revoke("unowned geometry or learner mutation")
        valid = (self.enabled and self.learner._game == self.game
                and getattr(self.calib, "_geometry_owner", self) is self
                and getattr(self.learner, "_geometry_owner", self) is self
                and (self.flow is None or (self.flow.calib is self.calib
                    and self.flow.lc is self.learner and self.flow.game == self.game
                    and self.flow.model_identity == self.model)))
        if not valid:
            self.enabled = False
            self.revoke("pair or model changed")
        return valid and fingerprint is not None and fingerprint == self._boundary

    def _input(self):
        geometry = asdict(self.calib)
        for name in ("pitch_deg", "yaw_deg", "height_m"):
            geometry.pop(name)
        return (geometry, self.learner.reset_generation, self.model)

    def after_reset(self):
        lc = self.learner
        # This hook runs only after the real reset and successful disk barrier.
        if (not self.enabled or not getattr(lc, "_geometry_reset_ok", False)
                or lc.samples or lc.blocks or lc._block_n or lc.bootstrap_samples
                or lc._heights or lc._bootstrap_height_n or lc._wide_n
                or lc._block_rpy_sum.any() or lc._block_height_sum
                or lc._bootstrap_rpy_sum.any() or lc._bootstrap_height_sum
                or lc._wide_sum.any() or lc.wide_from_device_estimate is not None
                or lc._accept_timestamps or lc._consecutive_invalid_blocks):
            return
        candidate = self._epoch_id_factory()
        if (type(candidate) is not str or not _HEX32.fullmatch(candidate)
                or candidate in self._issued):
            self.revoke("invalid or repeated measurement epoch")
            return
        self._issued.add(candidate)
        self.epoch = candidate
        self._epoch_input = self._input()
        self._fresh_samples = 0

    def after_update(self):
        from simsteer.core.learners.livecalib import CalStatus
        lc = self.learner
        if lc.cal_status == CalStatus.INVALID or (lc.cal_status == CalStatus.CALIBRATED
                                                  and not lc.writes_enabled):
            self.revoke("camera writes disabled or invalid")
        if self.epoch is None:
            return
        if self._input() != self._epoch_input:
            self.revoke("measurement inputs changed")
            return
        if self._fresh_samples and lc.writes_enabled and (lc._block_n == 0 or not self.trusted):
            self._boundary = self._fingerprint()
            self.publish()

    def _strict_pair(self, calibration, learner):
        from simsteer.core.calibration import Calibration, SIMPLE_WARP_VERSION
        from simsteer.core.learners.livecalib import (
            LiveCalib, CalStatus, CERTAINTY_STATE_VERSION, BLOCK_SIZE,
            INPUTS_WANTED, INPUTS_NEEDED, CERTAINTY_BOOTSTRAP_BLOCKS, blocked_counts_from_state,
        )
        import numpy as np
        if set(calibration) != {f.name for f in fields(Calibration)}:
            raise ValueError("incomplete camera geometry")
        # dataclass_values is intentionally more permissive for ordinary loads.
        for field in fields(Calibration):
            value = calibration[field.name]
            default = getattr(Calibration(), field.name)
            expected = type(default)
            if expected is float:
                state.bounded_number(value, -1e12, 1e12)
            elif type(value) is not expected:
                raise ValueError("invalid camera type")
        if (type(calibration["simple_warp_version"]) is not int
                or calibration["simple_warp_version"] != SIMPLE_WARP_VERSION):
            raise ValueError("camera algorithm not current")
        detached = Calibration(**calibration)
        detached.validate()
        test = LiveCalib()
        if set(learner) != set(test.state_dict()):
            raise ValueError("incomplete learner")
        if type(learner["certainty_state_version"]) is not int or learner["certainty_state_version"] != CERTAINTY_STATE_VERSION:
            raise ValueError("invalid certainty provenance")
        for name in ("bootstrap_samples", "bootstrap_height_n", "reset_generation", "wide_n",
                     "block_n", "samples", "rejected", "cal_status"):
            if type(learner[name]) is not int:
                raise ValueError("invalid learner count type")
            state.bounded_int(learner[name])
        if not (0 <= learner["block_n"] < BLOCK_SIZE
                and INPUTS_NEEDED * BLOCK_SIZE <= learner["bootstrap_samples"] <= CERTAINTY_BOOTSTRAP_BLOCKS * BLOCK_SIZE
                and learner["bootstrap_height_n"] <= learner["bootstrap_samples"]
                and learner["samples"] >= len(learner["rpys"]) * BLOCK_SIZE + learner["block_n"]):
            raise ValueError("inconsistent learner counts")
        for name in ("rpys", "heights"):
            if type(learner[name]) is not list or len(learner[name]) > INPUTS_WANTED:
                raise ValueError("invalid history")
        if len(learner["heights"]) > len(learner["rpys"]):
            raise ValueError("inconsistent height history")
        for rpy in learner["rpys"]:
            if type(rpy) is not list or len(rpy) != 3:
                raise ValueError("invalid pose vector")
            for angle in rpy:
                state.bounded_number(angle, -3.141592653589793, 3.141592653589793)
        for height in learner["heights"]:
            state.bounded_number(height, .0001, 100)
        mapping = {"bootstrap_samples": "bootstrap_samples", "bootstrap_rpy_sum": "_bootstrap_rpy_sum",
                   "bootstrap_height_sum": "_bootstrap_height_sum", "bootstrap_height_n": "_bootstrap_height_n",
                   "reset_generation": "reset_generation", "rpys": "_rpys", "heights": "_heights",
                   "wide_sum": "_wide_sum", "wide_n": "_wide_n", "wide_from_device_estimate": "wide_from_device_estimate",
                   "block_rpy_sum": "_block_rpy_sum", "block_height_sum": "_block_height_sum",
                   "block_n": "_block_n", "samples": "samples", "rejected": "rejected",
                   "last_rejection_reason": "last_rejection_reason"}
        for key, attr in mapping.items():
            value = learner[key]
            if key in ("bootstrap_rpy_sum", "wide_sum", "block_rpy_sum", "wide_from_device_estimate") and value is not None:
                if type(value) is not list or len(value) != 3:
                    raise ValueError("invalid learner vector")
                for number in value:
                    state.bounded_number(number, -1e12, 1e12)
                value = np.asarray(value, dtype=np.float64)
            elif key == "rpys":
                value = [np.asarray(r, dtype=np.float64) for r in value]
            setattr(test, attr, value)
        for name in ("block_height_sum", "bootstrap_height_sum"):
            state.bounded_number(learner[name], 0, 100 * CERTAINTY_BOOTSTRAP_BLOCKS * BLOCK_SIZE)
        test.blocked_counts = blocked_counts_from_state(learner["blocked_counts"])
        test.validate_state()
        test._refresh_cal_status()
        if (test.cal_status != CalStatus.CALIBRATED or not test.writes_enabled
                or learner["cal_status"] != int(test.cal_status)
                or test.state_dict() != learner
                or detached.pitch_deg != test.pitch_estimate
                or detached.yaw_deg != test.yaw_estimate
                or (test.height_estimate is not None and detached.height_m != test.height_estimate)):
            raise ValueError("camera is not the measured calibrated pair")
        return calibration, learner

    @staticmethod
    def _decode(raw):
        def reject(value):
            raise ValueError("invalid JSON constant")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        value = json.loads(raw, parse_constant=reject, object_pairs_hook=unique)
        if type(value) is not dict:
            raise ValueError("expected object")
        return value

    @staticmethod
    def _digest(raw):
        return {"sha256": hashlib.sha256(raw).hexdigest(), "length": len(raw)}

    def _read_pair(self):
        cal_path, lc_path, _ = self.paths
        for path in (cal_path, lc_path):
            state.flush_state(path, deadline=self.active_deadline)
        raw = (cal_path.read_bytes(), lc_path.read_bytes())
        pair = self._strict_pair(*(self._decode(value) for value in raw))
        if pair != self._capture():
            raise ValueError("physical pair differs from memory")
        return raw, pair

    def _make_receipt(self, raw, pair):
        from simsteer.core.calibration import SIMPLE_WARP_VERSION
        return {"schema": 1, "game": self.game, "simple_warp_version": SIMPLE_WARP_VERSION,
                "model": self.model, "epoch": self.epoch,
                "calibration": self._digest(raw[0]), "learner": self._digest(raw[1]),
                "geometry": pair[0]}

    def _valid_receipt(self, receipt, raw, pair):
        if type(receipt) is not dict or set(receipt) != {"schema", "game", "simple_warp_version", "model", "epoch", "calibration", "learner", "geometry"}:
            return False
        if (type(receipt["schema"]) is not int or receipt["schema"] != 1
                or type(receipt["simple_warp_version"]) is not int
                or type(receipt["epoch"]) is not str or not _HEX32.fullmatch(receipt["epoch"])):
            return False
        for name in ("calibration", "learner"):
            digest = receipt[name]
            if (type(digest) is not dict or set(digest) != {"sha256", "length"}
                    or type(digest["sha256"]) is not str or not _HEX64.fullmatch(digest["sha256"])
                    or type(digest["length"]) is not int
                    or not 0 < digest["length"] <= state.MAX_STATE_COUNT):
                return False
        self._strict_pair(receipt["geometry"], pair[1])
        expected = self._make_receipt(raw, pair)
        expected["epoch"] = receipt["epoch"]
        return receipt == expected

    def restore_receipt(self):
        try:
            cal_path, lc_path, receipt_path = self.paths
            if (getattr(self.calib, "_canonical_origin", None) != cal_path
                    or getattr(self.learner, "_canonical_origin", None) != lc_path):
                return False
            state.flush_state(receipt_path, deadline=self.active_deadline)
            receipt = self._decode(receipt_path.read_bytes())
            raw, pair = self._read_pair()
            if not self._valid_receipt(receipt, raw, pair):
                return False
            self.epoch = receipt["epoch"]
            self._issued.add(self.epoch)
            self._epoch_input = self._input()
            state._remaining(self.active_deadline)
            self._receipt, self.trusted = receipt, True
            return True
        except state.StateIOUnavailable:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self.revoke("startup receipt rejected")
            return False

    def publish(self):
        with self.operation():
            return self._publish()

    def _publish(self):
        if not self.audit():
            return False
        if self._publishing or self.epoch is None or not self._fresh_samples:
            return False
        self._publishing = True
        try:
            cal_path, lc_path, receipt_path = self.paths
            # A: drain the old receipt key before payloads enter the queue.
            state.flush_state(receipt_path, deadline=self.active_deadline)
            pair = self._strict_pair(*self._capture())
            fingerprint = self._fingerprint()
            # B: submission is only an ack; validate both physical files.
            for path, payload in zip((cal_path, lc_path), pair):
                if state.atomic_save_json(path, payload, deadline=self.active_deadline) is not True:
                    raise ValueError("payload was not submitted")
            raw, physical = self._read_pair()
            if physical != pair or self._fingerprint() != fingerprint:
                raise ValueError("pair changed during publication")
            receipt = self._make_receipt(raw, pair)
            # C: receipt last, with its own readback and final pair check.
            if state.atomic_save_json(receipt_path, receipt, deadline=self.active_deadline) is not True:
                raise ValueError("receipt was not submitted")
            state.flush_state(receipt_path, deadline=self.active_deadline)
            saved = self._decode(receipt_path.read_bytes())
            final_raw, final_pair = self._read_pair()
            if (saved != receipt or not self._valid_receipt(saved, final_raw, final_pair)
                    or self._fingerprint() != fingerprint or self._input() != self._epoch_input):
                raise ValueError("receipt acknowledgement failed")
            state._remaining(self.active_deadline)
            self._receipt, self.trusted = receipt, True
            return True
        except state.StateIOUnavailable:
            self.revoke("publication unavailable")
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self.revoke("publication rejected")
            return False
        finally:
            self._publishing = False

    def can_preserve_legacy_device(self, *, shared_version, device_version):
        with self.operation():
            return self._can_preserve_legacy_device(shared_version, device_version)

    def _can_preserve_legacy_device(self, shared_version, device_version):
        from simsteer.core.calibration import SIMPLE_WARP_VERSION
        from simsteer.core.learners.livecalib import CalStatus
        if (type(shared_version) is not int or shared_version != SIMPLE_WARP_VERSION
                or (type(device_version) is int and device_version == SIMPLE_WARP_VERSION)
                or not self.audit() or not self.trusted or self.epoch is None
                or self._input() != self._epoch_input):
            return False
        try:
            raw, pair = self._read_pair()
            receipt_path = self.paths[2]
            state.flush_state(receipt_path, deadline=self.active_deadline)
            receipt = self._decode(receipt_path.read_bytes())
            self._internal_read += 1
            try:
                self.learner._refresh_cal_status()
            finally:
                self._internal_read -= 1
            if (receipt != self._receipt or not self._valid_receipt(receipt, raw, pair)
                    or self.learner.cal_status != CalStatus.CALIBRATED
                    or not self.learner.writes_enabled or not self.audit()):
                raise ValueError("consume proof rejected")
            state._remaining(self.active_deadline)
            return True
        except state.StateIOUnavailable:
            self.revoke("consume unavailable")
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self.revoke("consume proof rejected")
            return False
