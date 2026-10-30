"""The holographic engine, vendored from itpartypattaya/hermes-plugin-holographic v0.25.2 (9dc7b768dc5284b2c8bfaccd6af2dac4a356a392) by tools/sync_engine.py.

Do not edit the modules here: change the plugin and vendor its next tag. This file is the only one written by the
sync; it supplies the Hermes core helpers the store calls when Hermes core is not installed (holomem.compat)."""

from .. import compat as _compat

_compat.install()
