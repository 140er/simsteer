"""Small loop adapters for optional physical-wheel sessions."""
from contextlib import contextmanager

from simsteer.core.override import BRAKE_TAKEOVER


@contextmanager
def output_guard(pad):
    """Also close a relay when capture fails before entering the loop."""
    try:
        yield
    finally:
        if getattr(pad, "physical", None) is not None:
            pad.close()


def player_axis(pad):
    p = getattr(pad, "physical", None)
    return p.steer if p and p.ok else (float("nan") if p else None)


def engage_block(pad):
    p = getattr(pad, "physical", None)
    if p is not None:
        if not p.ok:
            return "wheel not found"
        if p.brake is None:
            return "check pedals" if getattr(pad, "pedal_status", "") == "Check pedals" else "map pedals"
        if (p.brake or 0) >= BRAKE_TAKEOVER:
            return "release the brake"
    return ""


def driver(pad):
    p = getattr(pad, "physical", None)
    return ("No wheel" if not p.ok else "AI" if pad.engaged else "Player") if p else None


def log_fields(pad, takeover_reason, user_steer=None):
    p = getattr(pad, "physical", None)
    if p is None:
        return {}
    stats = pad.stats(consume=True)
    return dict(source=pad.source, rim_axis=p.steer if p.ok else None, user_steer=user_steer,
                phys_throttle=p.throttle, phys_brake=p.brake, takeover_reason=takeover_reason,
                relay_hz=stats["relay_hz"], relay_gap_max_ms=stats["relay_gap_max_ms"])


def dev_line(pad, rim_range_deg):
    p = getattr(pad, "physical", None)
    if p is None:
        return ""
    s = pad.stats()
    return (f"WHEEL {s['relay_hz']:.0f} Hz p95 {s['latency_p95_ms']:.1f} ms  "
            f"rim {p.steer * rim_range_deg / 2:+.1f}°  thr {p.throttle or 0:.2f} "
            f"brk {p.brake or 0:.2f}  src {pad.source}  {s['pedal_status']}"
            + (" LOW" if s['relay_hz'] < 250 else ""))
