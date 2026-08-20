# SimSteer v0.11.1 Test Plan

## Overview
This test plan covers the reliability, usability, and Fanatec support improvements for SimSteer.

## Critical Changes Summary
1. **Fanatec Wheel Support** - New output device type with DirectInput detection
2. **Enhanced Preflight Checks** - Detailed error messages with actionable fixes
3. **Auto-FOV Detection** - Enabled by default to fix #1 failure mode
4. **Improved Setup Wizard** - Clearer status messages and visual indicators

---

## Test Environment Requirements

### Hardware/Software
- Windows 10/11 PC
- Python 3.11 (for source testing)
- One of: ETS2, Forza Horizon 5, or Assetto Corsa
- **Ideal**: Access to a Fanatec wheel (CSL/ClubSport/Podium)
- **Minimum**: vJoy and ViGEm drivers for testing virtual output

### Test Data
- Fresh SimSteer installation (no previous calibration data)
- Various FOV settings in-game for auto-FOV testing

---

## 1. Fanatec Support Tests

### 1.1 Fanatec FFB Motor Detection (Ideal: Real Hardware)
**Prerequisites**: Fanatec wheel connected, driver installed, wheel in PC mode

**Steps**:
1. Launch SimSteer with `--device fanatec`
2. Check console output for mode
3. Check preflight dialog

**Expected**:
- ✓ Console shows: "Fanatec FFB motor control (physical wheel drive)" OR
- ⚠️ Console shows: "Fanatec mode (vJoy fallback - FFB unavailable)"
- ✓ Preflight info confirms detection
- ✓ Application launches successfully
- ✓ If FFB mode: Check console for background exclusive acquisition status

### 1.2 Fanatec FFB Motor Control (REQUIRES HARDWARE)
**Prerequisites**: Fanatec wheel in FFB motor mode (console confirms)

**Steps**:
1. Complete calibration with `--device fanatec`
2. Engage on highway with hands OFF wheel
3. Observe physical wheel rim

**Expected**:
- ✓ **Physical wheel rim TURNS by itself** to match AI steering
- ✓ Wheel moves smoothly left/right during lane keeping
- ✓ Motor force proportional to steering angle
- ✓ Game vehicle follows the physical wheel position
- ✓ No "wheel fighting" or oscillation

### 1.3 Fanatec Motor Release on Disengage
**Prerequisites**: Fanatec in FFB motor mode, currently engaged

**Steps**:
1. While engaged with AI steering, press INSERT to disengage
2. Immediately try to turn wheel manually

**Expected**:
- ✓ Motor releases immediately (wheel becomes easy to turn)
- ✓ No resistance from AI control
- ✓ Manual steering works normally
- ✓ Game responds to manual input

### 1.4 Fanatec FFB Fallback Behavior
**Prerequisites**: Fanatec wheel, but game has exclusive FFB lock OR FFB init fails

**Steps**:
1. Launch game first (some games take exclusive FFB)
2. Launch SimSteer with `--device fanatec`
3. Check console/preflight messages

**Expected**:
- ⚠️ Console: "Fanatec mode (vJoy fallback - FFB unavailable)"
- ✓ Preflight explains why FFB unavailable
- ✓ vJoy fallback works (virtual output)
- ✓ Application doesn't crash
- ✓ Can still use Fanatec for manual input, vJoy for AI output

### 1.5 Fanatec FFB Re-Acquisition After Conflict
**Prerequisites**: Fanatec in FFB mode, game takes/releases exclusive access

**Steps**:
1. Engage with Fanatec FFB mode
2. Alt-tab to game menu (game may reclaim exclusive FFB)
3. Return to driving
4. Check if motor control recovers

**Expected**:
- ✓ SimSteer detects DIERR_NOTACQUIRED gracefully
- ✓ Attempts re-acquisition after loss
- ✓ Either recovers FFB or falls back to vJoy smoothly
- ✓ No crashes or hangs

### 1.6 Fanatec Detection (Without Hardware)
**Prerequisites**: No Fanatec wheel connected

**Steps**:
1. Launch SimSteer with `--device fanatec`
2. Check preflight dialog

**Expected**:
- ⚠️ Warning: "No Fanatec wheel detected"
- ✓ Clear troubleshooting steps (driver, PC mode, USB)
- ✓ Console: "vJoy fallback" mode active
- ✓ Application launches with vJoy output

### 1.7 Fanatec vJoy Fallback Functionality
**Prerequisites**: vJoy driver installed, device #1 enabled, FFB unavailable

**Steps**:
1. Launch with `--device fanatec` (FFB fails to init)
2. Complete calibration
3. Engage on highway

**Expected**:
- ✓ Steering commands sent to vJoy device
- ✓ In-game vehicle responds to AI steering
- ✓ Physical Fanatec wheel stays neutral (no motor drive)
- ✓ User can manually steer with Fanatec
- ✓ Game sees both wheels (bind accordingly)

### 1.8 Fanatec in Setup Tab
**Steps**:
1. Launch SimSteer (any device mode)
2. Open tuner → Setup tab
3. Change device to "Fanatec"
4. Click "Save & Restart"

**Expected**:
- ✓ Fanatec appears in device dropdown
- ✓ Application restarts with Fanatec mode
- ✓ Setting persists across restarts
- ✓ Console shows FFB motor or vJoy fallback status

---

## 2. Preflight Check Tests

### 2.1 DirectML Missing Warning
**Prerequisites**: Remove or rename onnxruntime-directml

**Steps**:
1. `pip uninstall onnxruntime-directml`
2. Launch SimSteer

**Expected**:
- ⚠️ Prominent warning dialog: "DirectML unavailable — vision will run on CPU (VERY SLOW)"
- ✓ Performance impact explained (4-8 FPS vs 20+)
- ✓ Clear fix: "pip install onnxruntime-directml"
- ✓ Explains engagement gate will block at low FPS

### 2.2 ETS2 Deadzone Warning
**Prerequisites**: ETS2 installed, deadzone > 0 in config

**Steps**:
1. Set ETS2 steering deadzone to 16% in-game
2. Launch SimSteer with ETS2 running

**Expected**:
- ⚠️ Warning: "ETS2: steering deadzone is 16% — MUST BE 0"
- ✓ Emphasizes: "TRUCK WILL NOT STEER with deadzone > 0"
- ✓ Step-by-step fix in dialog
- ✓ Config file path shown

### 2.3 ETS2 SCS Plugin Missing
**Prerequisites**: ETS2 installed, no scs-telemetry.dll plugin

**Steps**:
1. Remove scs-telemetry.dll from ETS2 plugins folder
2. Launch SimSteer

**Expected**:
- ⚠️ Warning: "ETS2: SCS Telemetry plugin not installed"
- ✓ Clear warning: "ETS2 WILL NOT WORK without the telemetry plugin"
- ✓ [Install] button if bundled plugin available
- ✓ Manual install instructions with FROM/TO paths

### 2.4 vJoy Missing (Wheel Mode)
**Prerequisites**: No vJoy driver installed

**Steps**:
1. Uninstall vJoy driver
2. Launch SimSteer with `--device wheel`

**Expected**:
- ❌ Fatal error: "vJoy driver / pyvjoy not detected"
- ✓ Clear instructions: install vJoy + configure device #1
- ✓ Application doesn't launch (correct behavior)

### 2.5 FOV Reminder Warning
**Steps**:
1. Launch SimSteer with any game
2. Check preflight warnings

**Expected**:
- ⚠️ Warning present: "CRITICAL: Set camera FOV to match your in-game setting"
- ✓ Explains FOV is #1 cause of veering
- ✓ Provides step-by-step FOV setup
- ✓ Mentions auto-FOV will measure automatically
- ✓ Includes verification steps (ratio ~1.00)

---

## 3. Auto-FOV Detection Tests

### 3.1 Auto-FOV Enabled by Default
**Steps**:
1. Fresh install (delete settings.json)
2. Launch SimSteer
3. Check console output

**Expected**:
- ✓ Console shows: "auto-FOV: enabled (will measure on highway)"
- ✓ No `--no-auto-fov` flag needed
- ✓ `settings.auto_fov` defaults to True

### 3.2 Auto-FOV Measurement
**Prerequisites**: ETS2 or any game, highway driving available

**Steps**:
1. Set in-game FOV to 90 degrees
2. Set SimSteer Capture VFOV to 60 degrees (intentionally wrong)
3. Launch, complete camera calibration
4. Drive straight on highway at 15+ mph for 60 seconds
5. Watch HUD "auto-FOV" line

**Expected**:
- ✓ HUD shows: "AUTO-FOV measuring X/60"
- ✓ Progress increases during straight driving
- ✓ Pauses when turning or slowing down
- ✓ After 60 samples: Banner appears "auto-FOV: 60 → 90deg (vx/v_ego=X.XX)"
- ✓ FOV corrected and saved to calibration file
- ✓ Plan now tracks lane correctly

### 3.3 Auto-FOV One-Shot Behavior
**Steps**:
1. Complete auto-FOV detection (see 3.2)
2. Continue driving for another 10 minutes

**Expected**:
- ✓ HUD shows: "AUTO-FOV FOV 90deg (auto)"
- ✓ No further measurements taken
- ✓ Status stays "frozen" / done
- ✓ FOV doesn't change from saved value

### 3.4 Disable Auto-FOV
**Steps**:
1. Launch with `--no-auto-fov`
2. Check console and HUD

**Expected**:
- ✓ Console: "auto-FOV: disabled"
- ✓ HUD: "AUTO-FOV off"
- ✓ Manual FOV tuning required

---

## 4. Setup Wizard UI Tests

### 4.1 Camera Calibration Phase
**Steps**:
1. Fresh calibration (delete wizard flags)
2. Launch SimSteer
3. Drive on highway

**Expected Banner**:
- ✓ "SETUP — X% — Drive manually on highway (speed 15+ mph, straight line)"
- ✓ Blue color (80, 200, 255)
- ✓ ETA shown when available

**Expected Hints**:
- When too slow (< 15 mph):
  - ✓ "⚠️ TOO SLOW — speed up to 15+ mph for calibration (currently X mph)"
- When steering too much:
  - ✓ "⚠️ STEERING TOO MUCH — drive straight with gentle inputs while calibrating"
- When good:
  - ✓ "✓ Good — keep driving on highway (~Xmin left @ X samples/s)"
  - OR "✓ Calibrating — keep driving straight on highway"

### 4.2 Steering Calibration Phase
**Steps**:
1. Complete camera calibration
2. Check banner and hints

**Expected**:
- ✓ Banner: "SETUP — X% — Press INSERT to engage, drive gently"
- ✓ Cyan color (80, 220, 200)
- ✓ Hint: "✓ Camera done! Press INSERT to engage and drive gently"

### 4.3 Ready State
**Steps**:
1. Complete both calibration phases
2. Watch for ready transition

**Expected**:
- ✓ Banner: "✓ READY — Press INSERT to engage autonomous driving" (10 seconds)
- ✓ Green color (80, 255, 80)
- ✓ Ready chime plays (if audio files present)
- ✓ Then collapses to: "✓ READY"
- ✓ Hint: "✓ Setup complete — ready to drive"

### 4.4 Calibration Reset (R key)
**Steps**:
1. During/after calibration, press R key

**Expected**:
- ✓ Hint immediately shows: "⚠️ CALIBRATION RESET: [reason]"
- ✓ Progress bar returns to 0%
- ✓ Wizard restarts from CAMERA phase
- ✓ Banner shows "SETUP — 0% — Drive manually..."

---

## 5. Integration Tests

### 5.1 First-Time User Flow (Gamepad)
**Steps**:
1. Fresh Windows VM or clean test environment
2. Install Python 3.11, requirements.txt
3. Install ViGEm driver
4. Install ETS2, configure Data Out
5. Launch SimSteer

**Expected Journey**:
1. ✓ Preflight shows all requirements clearly
2. ✓ FOV reminder appears prominently
3. ✓ ETS2 deadzone warning if not 0
4. ✓ Setup wizard guides through calibration
5. ✓ Auto-FOV measures and corrects FOV automatically
6. ✓ Ready banner appears, engagement works
7. ✓ Truck steers autonomously on highway

**Time to First Engage**: Should be < 20 minutes (down from 30-40 min)

### 5.2 First-Time User Flow (Fanatec)
**Prerequisites**: Fanatec wheel available

**Steps**:
1. Fresh install
2. Install Fanatec driver, vJoy, requirements
3. Wheel in PC mode
4. Launch SimSteer with `--device fanatec`

**Expected**:
1. ✓ Fanatec detected in preflight
2. ✓ Setup wizard completes normally
3. ✓ vJoy output works for AI steering
4. ✓ User's Fanatec still available for manual control
5. ✓ Game sees both wheels (user can bind both)

### 5.3 Existing User Upgrade
**Prerequisites**: Existing SimSteer v0.11.0 installation

**Steps**:
1. Keep existing calibration files
2. Upgrade to v0.11.1
3. Launch

**Expected**:
- ✓ Existing calibration loads
- ✓ Auto-FOV respects existing FOV (doesn't re-run if done=True)
- ✓ Settings.json migrates cleanly (new fields default)
- ✓ No breaking changes to saved data

### 5.4 Engagement Gate Validation
**Steps**:
1. Try to engage before calibration complete
2. Try to engage with low FPS
3. Try to engage without telemetry

**Expected**:
- ✓ Engage blocked with clear reason shown in UI
- ✓ Console shows specific gate failure
- ✓ Preflight warnings guide user to fix

---

## 6. Error Recovery Tests

### 6.1 Auto-FOV Bad Measurement
**Steps**:
1. Set FOV to 90 degrees
2. During auto-FOV measurement, drive in circles (simulate bad data)
3. Check if FOV goes to nonsense value

**Expected**:
- ✓ FovResolver clamps to [40, 150] degrees
- ✓ Bad samples filtered by yaw rate gate
- ✓ Takes median of 60 samples (outlier rejection)
- ✓ Resulting FOV is reasonable

### 6.2 Preflight Dialog Handling
**Steps**:
1. Trigger multiple warnings (no plugin, deadzone>0, no DirectML)
2. Check dialog behavior

**Expected**:
- ✓ Fatal errors shown first, stop launch
- ✓ Warnings allow continuation
- ✓ Install actions work (SCS plugin button)
- ✓ All errors readable (not truncated)

### 6.3 Device Fallback
**Steps**:
1. Set device to "fanatec" in settings
2. Uninstall vJoy
3. Launch

**Expected**:
- ❌ Fatal error with clear message
- ✓ Explains vJoy required for Fanatec mode
- ✓ Doesn't crash, clean error handling

---

## 7. Regression Tests

### 7.1 Existing Gamepad Mode
**Steps**:
1. Use default gamepad mode
2. Complete full drive cycle

**Expected**:
- ✓ No breaking changes
- ✓ ViGEm output works identically
- ✓ All existing features work
- ✓ Performance unchanged

### 7.2 Existing Wheel (vJoy) Mode
**Steps**:
1. Use `--device wheel`
2. Complete full drive cycle

**Expected**:
- ✓ vJoy output unchanged
- ✓ Linear steering response preserved
- ✓ All existing features work

### 7.3 Manual FOV Adjustment
**Steps**:
1. Disable auto-FOV with `--no-auto-fov`
2. Manually tune FOV in tuner

**Expected**:
- ✓ Manual FOV slider still works
- ✓ FOV ratio in HUD updates correctly
- ✓ No interference from auto-FOV system

---

## 8. Documentation Tests

### 8.1 README Accuracy
**Steps**:
1. Follow README instructions exactly
2. Verify all links work
3. Check troubleshooting section

**Expected**:
- ✓ Installation steps accurate
- ✓ Fanatec section clear
- ✓ Auto-FOV documented
- ✓ All links valid

### 8.2 SETUP.md Walkthrough
**Steps**:
1. Fresh user follows SETUP.md step-by-step
2. Complete first drive

**Expected**:
- ✓ All prerequisites listed
- ✓ Fanatec setup clear
- ✓ Auto-FOV section accurate
- ✓ Troubleshooting helpful

---

## 9. Performance Tests

### 9.1 Preflight Performance
**Steps**:
1. Launch SimSteer
2. Measure preflight check time

**Expected**:
- ✓ Preflight completes < 2 seconds
- ✓ No noticeable delay vs v0.11.0
- ✓ Fanatec detection doesn't hang

### 9.2 Auto-FOV Performance
**Steps**:
1. Monitor FPS during auto-FOV measurement

**Expected**:
- ✓ No FPS impact (calculations are lightweight)
- ✓ Vision/control loop unchanged

---

## 10. Known Limitations

### Cannot Test Without Hardware
- **Real Fanatec FFB motor behavior**: Can only verify architecture + fallback
- **Motor torque / force calibration**: Cannot tune without hardware feedback
- **FFB conflict resolution**: Game/SimSteer exclusive access race conditions
- **DirectInput ctypes implementation**: Full Win32 API calls need testing

### Acceptable Behavior
- Fanatec FFB mode falls back to vJoy if DirectInput acquisition fails
- Current implementation documents FFB architecture but may return False from
  `_init_fanatec_ffb()` until Win32 DirectInput ctypes bindings are complete
- Auto-FOV requires ~60 seconds straight driving (not instant)
- Preflight warnings always show FOV reminder (by design)

---

## Success Criteria

### Critical (Must Pass)
- [ ] All 3 device modes launch and output correctly
- [ ] Preflight warnings show for all error conditions
- [ ] Auto-FOV detects and corrects FOV within 2 minutes of highway driving
- [ ] Setup wizard shows clear, actionable messages
- [ ] No regressions in existing gamepad/wheel modes
- [ ] Documentation matches implementation

### Important (Should Pass)
- [ ] Fanatec detection works when wheel connected
- [ ] Fanatec fallback works without wheel
- [ ] All preflight error messages include fix instructions
- [ ] Auto-FOV can be disabled
- [ ] First-time setup < 20 minutes

### Nice-to-Have
- [ ] Fanatec + real wheel coexistence verified
- [ ] Auto-FOV works across all 3 games
- [ ] Zero questions in GitHub issues about FOV setup

---

## Test Execution Notes

Record results for each test:
- ✓ Pass
- ❌ Fail
- ⚠️ Partial (document why)
- ⏭️ Skip (document reason, e.g., "no hardware")

For failures, create GitHub issues with:
- Test case ID
- Steps to reproduce
- Expected vs actual behavior
- Logs/screenshots
