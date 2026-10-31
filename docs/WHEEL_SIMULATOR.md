# Testing without a wheel

`simsteer.testing.fanatec` simulates a Fanatec base behind SimSteer's real
DirectInput and vJoy code. It plugs in where SimSteer calls the Windows
libraries (`dinput_out._com_call` and `vjoy_ffb.load_dll`), so detection,
data formats, acquire and loss, the force feedback sink and the FFB bridge
run unchanged, on Linux or Windows, with no driver and no device.

Only tests and the command below install it; the app never does.

## What it imitates

- **Podium DD2** (`profile="dd2"`): USB 0EB7:0007, enumerated as two
  DirectInput instances with the same product name (8 axes / 108 buttons
  and 12 axes / 63 buttons), one force feedback actuator on "Wheel Axis"
  (object ID 0x01000002) and the eleven standard effects plus CustomForce.
  `live_instance` picks which instance carries input (0, 1 or "both").
- Also `single` (one instance), `two_bases` and `none` (only vJoy).
- DirectInput rules: data format before Acquire, exclusive access for
  force feedback, one exclusive owner per base, `DIERR_INPUTLOST` once and
  then `DIERR_NOTACQUIRED` after an unplug, new device objects after a
  replug. DD2 quirks: autocentre only before Acquire, effects must name the
  axis by object ID.
- vJoy: the setup check, feeding axes and buttons, and the game's force
  feedback packets delivered to SimSteer's callback.

## Python API

```python
from simsteer.testing.fanatec import FanatecSim, Track

with FanatecSim(profile="dd2", live_instance=1) as sim:
    sim.wheel.set(steer_deg=0, throttle=0, brake=0)
    sim.wheel.play(Track().hold(1).ramp(steer_deg=12.5, over=0.08)
                   .noise(seconds=60, steer_deg=2, seed=7)
                   .pedal("brake", 0.12, hold=0.06)
                   .unplug(for_s=1.5))
    sim.wheel.play(Track.from_simsteer_jsonl("run.jsonl"))  # replay a logged drive
    sim.game.send("recorded_sender_default", realtime=True)  # the game's FFB packets
    sim.game.axes                                            # what the game sees on vJoy

    sim.ffb.assert_force_never_above(0.15)
    sim.ffb.assert_gain_set_before_first_start(cap=0.05)
    sim.ffb.assert_all_stopped(after=t_stop, within_ms=200)
    sim.ffb.assert_quiet_after(t_stop)
    sim.ffb.assert_no_force_writes()      # read-only paths
    sim.assert_no_leaks()                 # every COM object released
    sim.assert_single_thread()
```

Faults: `sim.wheel.unplug()` / `replug()`, `unplug_pedals()`,
`hold_exclusive()` (a game owns the base), `emergency_stop()` (safety
switch), `refuse_gain = lambda value: ...`, `fail(method, hresult)`.

The force figures are an upper bound of the commanded torque as a share of
the base's peak (effect magnitudes times effect gain, summed and clipped,
times the device gain). An effect with a duration keeps playing until it
ends even if the process that started it dies, the worst case the
bridge's renewal windows are built for.

## Command line

```
python -m simsteer.testing.fanatec enumerate [--profile dd2] [--live 1]
python -m simsteer.testing.fanatec bridge --cap 5 --packets recorded_sender_default --events ev.jsonl
python -m simsteer.testing.fanatec report ev.jsonl --cap 5 [--killed-at T --within-ms 250]
```

`bridge` runs the real vJoy receiver, FFB model and DirectInput sink
against the simulator and streams every force event to a file as it
happens, so a run that is killed can still be judged by `report`.

## Limits

The DD2's slot layout after SimSteer's data format, which instance carries
live input, the pedals' rest direction and the driver's behaviour when a
process is killed are taken from one enumeration or assumed worst case;
check them on a real base when they matter.
