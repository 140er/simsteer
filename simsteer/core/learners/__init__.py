"""Online learners ported from openpilot: LiveParams (steering-rack RLS) and
LiveCalib (camera-pose, openpilot's calibrationd). These are the ONLY things
that learn online — both observe signals independent of what they tune, which
is why they don't run away. Everything else is a static knob.
"""
