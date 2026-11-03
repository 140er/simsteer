"""Strict provenance, durable publication and serialized ownership boundaries."""
from dataclasses import asdict, replace
import copy
import json
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simsteer import state, state_import
from simsteer.calib import flow as flow_module
from simsteer.calib.flow import Step
from simsteer.core import calibration as calibration_module
from simsteer.core.calibration import Calibration
from simsteer.core.learners import livecalib as lc_module, liveparams as lp_module
from simsteer.core.learners.livecalib import CalStatus, LiveCalib
from simsteer.core.model_identity import pinned_model_identity
from simsteer.core.shared_geometry import SharedGeometry, bind_geometry
from simsteer.paths import state_path
from test_shared_camera_regression import fresh_camera, make_flow, measure, legacy_devices


def consume(owner):
    return owner.can_preserve_legacy_device(shared_version=2, device_version=1)


def assert_released(owner):
    acquired = threading.Event()
    def acquire():
        with owner.lock:
            acquired.set()
    thread = threading.Thread(target=acquire)
    thread.start()
    assert acquired.wait(2)
    thread.join(2)
    assert not thread.is_alive()


def trusted():
    current = fresh_camera()
    owner = current._geometry_owner
    assert owner.trusted and owner.epoch and consume(owner)
    return current, owner


def save_json(path, raw):
    path.write_text(json.dumps(raw, allow_nan=True))


def rewrite_receipt_digest(owner, index):
    path = owner.paths[index]
    receipt_path = owner.paths[2]
    receipt = json.loads(receipt_path.read_bytes())
    receipt["calibration" if index == 0 else "learner"] = owner._digest(path.read_bytes())
    save_json(receipt_path, receipt)


@pytest.mark.parametrize("field,value", [
    ("simple_warp_version", None), ("simple_warp_version", 1),
    ("simple_warp_version", 3), ("simple_warp_version", "2"),
    ("simple_warp_version", True), ("image_w", True), ("image_h", 0),
    ("fov_h_deg", "90"), ("fov_h_deg", 180.), ("height_m", -1.),
    ("pitch_deg", float("nan")), ("yaw_deg", float("inf")),
    ("crop_top_pct", .9), ("simple_warp", 1), ("cx_frac", None),
])
@pytest.mark.parametrize("rehash", [False, True])
def test_invalid_raw_calibration_never_preserves(field, value, rehash):
    current, owner = trusted()
    path = owner.paths[0]
    data = json.loads(path.read_bytes())
    if value is None:
        data.pop(field)
    else:
        data[field] = value
    save_json(path, data)
    if rehash:
        rewrite_receipt_digest(owner, 0)
    # In-memory current version cannot bless broken raw state.
    assert current.calib.simple_warp_version == 2
    assert not consume(owner)
    assert not owner.trusted and owner.epoch is None


@pytest.mark.parametrize("index,raw", [(0, b'{'), (0, b'[]'), (0, b''), (1, b'{'),
                                       (2, b'{'), (2, b'null')])
def test_missing_or_damaged_pair_is_rejected(index, raw):
    _, owner = trusted()
    owner.paths[index].write_bytes(raw)
    assert not consume(owner)
    assert not owner.trusted


@pytest.mark.parametrize("index", [0, 1, 2])
def test_missing_file_fails_closed(index):
    _, owner = trusted()
    owner.paths[index].unlink()
    assert not consume(owner)


@pytest.mark.parametrize("field,value", [
    ("schema", True), ("schema", 2), ("schema", "1"), ("epoch", ""),
    ("epoch", None), ("epoch", "A" * 32), ("epoch", "a" * 31),
    ("game", "other"), ("model", {}), ("simple_warp_version", True),
    ("simple_warp_version", 1), ("geometry", {}),
    ("calibration", {"sha256": "x" * 64, "length": 1}),
    ("learner", {"sha256": "a" * 64, "length": True}),
    ("learner", {"sha256": "a" * 64, "length": 0}),
    ("learner", {"sha256": "a" * 64, "length": 2**40}),
    ("learner", {"sha256": "a" * 64}),
])
def test_receipt_schema_is_strict(field, value):
    _, owner = trusted()
    path = owner.paths[2]
    data = json.loads(path.read_bytes())
    data[field] = value
    save_json(path, data)
    assert not consume(owner)


@pytest.mark.parametrize("key", ["schema", "game", "epoch", "geometry", "learner", "model"])
def test_missing_receipt_members_rejected(key):
    _, owner = trusted()
    receipt = json.loads(owner.paths[2].read_bytes())
    receipt.pop(key)
    save_json(owner.paths[2], receipt)
    assert not consume(owner)


def test_unknown_and_duplicate_receipt_members_rejected():
    _, owner = trusted()
    raw = owner.paths[2].read_bytes()
    owner.paths[2].write_bytes(raw[:-1] + b', "epoch": "' + b'a' * 32 + b'"}')
    assert not consume(owner)
    owner.paths[2].write_bytes(raw[:-1] + b', "unknown": 1}')
    resumed = make_flow()
    assert not resumed._geometry_owner.trusted


@pytest.mark.parametrize("field,value", [
    ("certainty_state_version", 0), ("certainty_state_version", True),
    ("cal_status", 0), ("cal_status", 2), ("cal_status", True),
    ("bootstrap_samples", 299), ("bootstrap_samples", 501),
    ("bootstrap_samples", "300"), ("bootstrap_height_n", 501),
    ("samples", 1), ("block_n", 100), ("rpys", [[0., 9., 0.]] * 3),
    ("rpys", [[0., .02]] * 3), ("heights", [-1.] * 3),
    ("block_rpy_sum", [float("nan"), 0., 0.]),
    ("wide_sum", [0., 0.]), ("reset_generation", True),
    ("blocked_counts", {"speed": True}),
])
def test_invalid_or_inconsistent_learner_cannot_be_trusted(field, value):
    _, owner = trusted()
    path = owner.paths[1]
    data = json.loads(path.read_bytes())
    data[field] = value
    save_json(path, data)
    rewrite_receipt_digest(owner, 1)
    assert not consume(owner)


@pytest.mark.parametrize("edit", ["fov", "vfov", "frame", "height", "pitch", "yaw",
                                  "crop-top", "crop-bottom", "mirror", "warp", "load",
                                  "rescale", "reset", "model", "writes", "invalid",
                                  "private-helper", "array", "deque", "bootstrap-array"])
def test_mutation_revokes_before_another_sample_can_adopt_old_history(edit, monkeypatch):
    current, owner = trusted()
    cal, lc = current.calib, current.lc
    actions = {
        "fov": lambda: setattr(cal, "fov_h_deg", 76.),
        "vfov": lambda: setattr(cal, "fov_v_deg", 45.),
        "frame": lambda: cal.update_for_frame((720, 1280)),
        "height": lambda: setattr(cal, "height_m", 1.5),
        "pitch": lambda: setattr(cal, "pitch_deg", 1.7),
        "yaw": lambda: setattr(cal, "yaw_deg", 1.7),
        "crop-top": lambda: setattr(cal, "crop_top_pct", .1),
        "crop-bottom": lambda: setattr(cal, "crop_bottom_pct", .1),
        "mirror": lambda: setattr(cal, "lateral_sign", 1.),
        "warp": lambda: setattr(cal, "simple_warp", False),
        "load": lambda: lc.load_state("ets2"),
        "rescale": lambda: lc.rescale_for_fov(cal, 90., 75.),
        "reset": lambda: lc.reset("test reset"),
        "model": lambda: current.model_identity.update(adapter_version=99),
        "writes": lambda: monkeypatch.setattr(lc_module, "WRITE_CALIBRATION", False),
        "invalid": lambda: setattr(lc, "cal_status", CalStatus.INVALID),
        "private-helper": lambda: lc._write_pose(cal),
        "array": lambda: lc._rpys[0].__setitem__(1, .025),
        "deque": lambda: lc._heights.append(1.5),
        "bootstrap-array": lambda: lc._bootstrap_rpy_sum.__setitem__(1, .12),
    }
    old_epoch = owner.epoch
    actions[edit]()
    measure(current, 1)
    assert not consume(owner)
    assert not owner.trusted
    assert owner.epoch is None or (edit == "reset" and owner.epoch != old_epoch)


def test_defaults_generic_fallback_and_old_history_do_not_mint_origin():
    current = fresh_camera()
    owner = current._geometry_owner
    owner.paths[2].unlink()
    reloaded = make_flow()
    assert reloaded._geometry_owner.epoch is None
    measure(reloaded, 1)
    reloaded.calib.save(game="ets2")
    reloaded.lc.save_state("ets2")
    assert not consume(reloaded._geometry_owner)
    # A canonical learner and a valid receipt cannot authorize fallback calibration.
    owner.paths[0].rename(state_path("calibration"))
    save_json(owner.paths[2], owner._receipt)
    fallback = make_flow()
    assert fallback._geometry_owner.epoch is None


def test_detached_dataclass_replacement_has_no_owner():
    current, owner = trusted()
    candidate = replace(current.calib, fov_h_deg=75.)
    assert "_geometry_owner" not in candidate.__dict__
    assert "_geometry_owner" not in asdict(current.calib)
    candidate.pitch_deg = 5.
    current.calib.fov_h_deg = current.calib.fov_h_deg
    assert consume(owner)
    current.calib.fov_h_deg += 1.
    assert not owner.trusted and owner.epoch is None
    assert not consume(owner)


@pytest.mark.parametrize("factory", [lambda: "", lambda: None, lambda: "X" * 32])
def test_invalid_factory_is_not_retried(factory):
    current, owner = trusted()
    calls = []
    owner._epoch_id_factory = lambda: calls.append(1) or factory()
    current.lc.reset()
    assert calls == [1]
    assert owner.epoch is None and not owner.trusted


def test_reset_ids_are_deterministic_unique_and_only_allocated_after_success(monkeypatch):
    current, owner = trusted()
    ids = iter(["a" * 32, "b" * 32, "a" * 32])
    owner._epoch_id_factory = lambda: next(ids)
    for expected in ("a" * 32, "b" * 32, None):
        current.lc.reset()
        assert owner.epoch == expected
    error = state.StateIOUnavailable("failed delete")
    def fail(*args, **kw):
        raise error
    owner._epoch_id_factory = lambda: pytest.fail("failed reset allocated an epoch")
    monkeypatch.setattr(lc_module, "delete_state", fail)
    with pytest.raises(state.StateIOUnavailable) as caught:
        current.lc.reset()
    assert caught.value is error
    assert owner.epoch is None
    assert_released(owner)


def test_none_factory_explicitly_uses_uuid(monkeypatch):
    current = make_flow()
    monkeypatch.setattr("simsteer.core.shared_geometry.uuid.uuid4", lambda: SimpleNamespace(hex="c" * 32))
    owner = SharedGeometry(current.calib, current.lc, "ets2", pinned_model_identity(), epoch_id_factory=None)
    assert owner._epoch_id_factory() == "c" * 32


@pytest.mark.parametrize("kind", ["model", "pair", "game", "nonpersistent", "default"])
def test_unsupported_or_conflicting_binding_disables_preservation(kind):
    current, owner = trusted()
    if kind == "model":
        make_flow(cal=current.calib, lc=current.lc).model_identity["adapter_version"] = 999
    elif kind == "pair":
        make_flow(cal=Calibration(), lc=current.lc)
    elif kind == "game":
        current.game = "other"
    elif kind == "nonpersistent":
        bind_geometry(current.calib, current.lc, "ets2", current.model_identity, persist=False, flow=current)
    else:
        bind_geometry(current.calib, current.lc, "default", current.model_identity, persist=True, flow=current)
    assert not consume(owner)
    current.lc.reset()
    assert owner.epoch is None


@pytest.mark.parametrize("case", ["lc-save", "lc-load", "lc-probe", "lc-reset", "cal-save",
                                  "cal-load", "cal-probe", "flow-save", "flow-load", "enter",
                                  "lp-reset", "lp-sign"])
def test_dedicated_error_escapes_bound_layers_and_releases_owner(case, monkeypatch):
    current, owner = trusted()
    error = state.StateIOUnavailable(case)
    seen = []
    def fail(*args, **kwargs):
        seen.append(kwargs.get("deadline"))
        raise error
    actions = {
        "lc-save": (lc_module, "atomic_save_json", lambda: current.lc.save_state("ets2")),
        "lc-load": (lc_module, "read_json_object", lambda: current.lc.load_state("ets2")),
        "lc-probe": (lc_module, "snapshot_exists", lambda: current.lc.load_state("ets2")),
        "lc-reset": (lc_module, "delete_state", current.lc.reset),
        "cal-save": (calibration_module, "atomic_save_json", lambda: current.calib.save(game="ets2")),
        "cal-load": (calibration_module, "read_json_object", lambda: Calibration.load("ets2", deadline=owner.active_deadline)),
        "cal-probe": (calibration_module, "snapshot_exists", lambda: Calibration.load("ets2", path=owner.paths[0], deadline=owner.active_deadline)),
        "flow-save": (flow_module, "atomic_save_json", current.save),
        "flow-load": (flow_module, "read_json_object", current._load),
        "enter": (flow_module, "atomic_save_json", lambda: current._enter(Step.CAMERA)),
        "lp-reset": (lp_module, "delete_state", current._reset_steering),
        "lp-sign": (lp_module, "atomic_save_json", lambda: current.lp.set_axis_sign(1, deadline=owner.active_deadline)),
    }
    module, name, action = actions[case]
    monkeypatch.setattr(module, name, fail)
    with pytest.raises(state.StateIOUnavailable) as caught:
        with owner.operation():
            deadline = owner.active_deadline
            action()
    assert caught.value is error and seen == [deadline]
    assert not owner.trusted and owner.epoch is None
    assert owner.active_deadline is None
    assert_released(owner)


@pytest.mark.parametrize("stage", range(1, 10))
def test_publication_barrier_exceptions_never_acknowledge(stage, monkeypatch):
    current = make_flow()
    current.lc.reset()
    measure(current, 299)
    owner = current._geometry_owner
    count = [0]
    original_flush, original_save = state.flush_state, state.atomic_save_json
    def boundary(function):
        def call(*args, **kw):
            count[0] += 1
            if count[0] == stage:
                raise OSError("injected publication fault")
            return function(*args, **kw)
        return call
    monkeypatch.setattr(state, "flush_state", boundary(original_flush))
    monkeypatch.setattr(state, "atomic_save_json", boundary(original_save))
    measure(current, 1)
    assert count[0] == stage
    assert not owner.trusted and owner.epoch is None
    assert not consume(owner)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("ack", [False, True, None])
def test_false_none_or_silent_submission_cannot_publish(index, ack, monkeypatch):
    current = make_flow()
    current.lc.reset()
    measure(current, 299)
    owner = current._geometry_owner
    original = state.atomic_save_json
    def save(path, payload, **kw):
        if path == owner.paths[index]:
            path.write_bytes(b'{}')
            return ack
        return original(path, payload, **kw)
    monkeypatch.setattr(state, "atomic_save_json", save)
    measure(current, 1)
    assert not owner.trusted and owner.epoch is None
    assert not consume(owner)


def test_background_writer_old_receipt_is_drained_before_new_pair(monkeypatch):
    current, owner = trusted()
    old_receipt = copy.deepcopy(owner._receipt)
    current.lc.reset()
    measure(current, 299)
    trace = []
    original = state._write_snapshot
    def write(path, payload):
        # No mutable owner enters the writer; owner is held in the producer.
        assert isinstance(payload, (str, bytes)) or payload is None
        trace.append(path.name)
        return original(path, payload)
    monkeypatch.setattr(state, "_write_snapshot", write)
    with state.background_saves():
        state.atomic_save_json(owner.paths[2], old_receipt)
        measure(current, 1)
        assert owner.trusted
        assert consume(owner)
    receipts = [i for i, path in enumerate(trace) if path == owner.paths[2].name]
    assert len(receipts) == 2
    for path in owner.paths[:2]:
        assert receipts[0] < max(i for i, name in enumerate(trace) if name == path.name) < receipts[-1]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_mixed_generation_crash_pair_rejected_on_startup(index):
    current, owner = trusted()
    older = [p.read_bytes() for p in owner.paths]
    current.lc.reset()
    measure(current, road_transform=np.array([0., 0., 1.6]))
    assert owner.trusted
    owner.paths[index].write_bytes(older[index])
    restarted = make_flow()
    assert not restarted._geometry_owner.trusted
    assert not consume(restarted._geometry_owner)


@pytest.mark.parametrize("model", [None, {}, {"adapter_version": True}, {"source_revision": "unknown"}])
def test_absent_or_unpinned_identity_is_not_implicitly_authorized(model):
    current = make_flow()
    detached_cal, detached_lc = Calibration(), LiveCalib(game="ets2")
    from simsteer.calib.flow import CalibFlow
    untrusted = CalibFlow("ets2", "wheel", calib=detached_cal, live_calib=detached_lc,
                          live_params=current.lp, model_identity=model)
    assert untrusted._geometry_owner is None
    detached_lc.reset()
    measure(untrusted)
    assert not state_path("shared_geometry", "ets2").exists()


@pytest.mark.parametrize("case", ["empty-epoch", "invalid-calibration", "old-learner", "no-origin"])
def test_startup_never_repairs_receipt_or_mints_new_epoch(case, monkeypatch):
    _, owner = trusted()
    if case == "empty-epoch":
        receipt = copy.deepcopy(owner._receipt)
        receipt["epoch"] = ""
        save_json(owner.paths[2], receipt)
    elif case == "invalid-calibration":
        data = json.loads(owner.paths[0].read_bytes())
        data["fov_h_deg"] = 190.
        save_json(owner.paths[0], data)
        rewrite_receipt_digest(owner, 0)
    elif case == "old-learner":
        data = json.loads(owner.paths[1].read_bytes())
        data.pop("certainty_state_version")
        save_json(owner.paths[1], data)
        rewrite_receipt_digest(owner, 1)
    monkeypatch.setattr("simsteer.core.shared_geometry.uuid.uuid4", lambda: pytest.fail("startup minted an epoch"))
    if case == "no-origin":
        cal = Calibration(**asdict(owner.calib))
        current = make_flow(cal=cal)
    else:
        current = make_flow()
    assert not current._geometry_owner.trusted
    assert current._geometry_owner.epoch is None


def test_failed_ordinary_delete_or_silent_failed_delete_cannot_start_epoch(monkeypatch):
    current, owner = trusted()
    def fail(*args, **kw):
        raise OSError("ordinary disk failure")
    monkeypatch.setattr(lc_module, "delete_state", fail)
    current.lc.reset()
    assert owner.epoch is None
    monkeypatch.setattr(lc_module, "delete_state", lambda *args, **kw: None)
    current.lc.reset()
    assert owner.epoch is None


@pytest.mark.parametrize("simple_warp", [True, False], ids=["simple", "perspective"])
def test_restart_keeps_baseline_reset_then_fov_geometry(simple_warp, monkeypatch):
    current, owner = trusted()
    current.calib.simple_warp = simple_warp
    geometry = asdict(current.calib)
    assert abs(geometry["pitch_deg"]) > .5 and abs(geometry["yaw_deg"]) > .25
    generation = current.lc.reset_generation
    current.step = Step.DONE
    current.result = {"passed": True}
    current.attempts = 3
    trace, reset_epochs = [], []
    reset, enter = current.lc.reset, current._enter

    def observed_reset(*args, **kwargs):
        trace.append(("reset", (current.calib.pitch_deg, current.calib.yaw_deg)))
        result = reset(*args, **kwargs)
        reset_epochs.append(owner.epoch)
        return result

    def observed_enter(step, *args, **kwargs):
        trace.append(("enter", step))
        return enter(step, *args, **kwargs)

    monkeypatch.setattr(current.lc, "reset", observed_reset)
    monkeypatch.setattr(current, "_enter", observed_enter)
    current.restart("explicit restart test")

    # The existing reset must see the measured pose; FOV entry owns zeroing.
    assert trace == [("reset", (geometry["pitch_deg"], geometry["yaw_deg"])),
                     ("enter", Step.FOV)]
    expected = geometry | ({"pitch_deg": 0., "yaw_deg": 0.} if simple_warp else {})
    assert asdict(current.calib) == expected
    assert current.lc.reset_generation == generation + 1
    assert current.lc.samples == current.lc.blocks == current.lc._block_n == 0
    assert current.lc.bootstrap_samples == 0
    assert current.lc.cal_status == CalStatus.UNCALIBRATED
    assert current.step is Step.FOV and current.result is None and current.attempts == 0
    assert not current.calibrated and not current.ai_steering_ready
    assert reset_epochs[0] is not None
    # Simple-warp FOV zeroing revokes the reset epoch; perspective keeps it pending.
    if simple_warp:
        assert owner.epoch is None
    else:
        assert owner.epoch == reset_epochs[0]
    assert not owner.trusted and not consume(owner)
    assert_released(owner)
