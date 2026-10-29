"""The holographic engine, vendored from itpartypattaya/hermes-plugin-holographic v0.24.0 (7adde2e4d80d4ba6fc655a0b04f07a0536b2cfbd) by tools/sync_engine.py.

Do not edit the modules here: change the plugin and vendor its next tag. This file is the only one written by the
sync; it supplies the Hermes core helpers the store calls when Hermes core is not installed (holomem.compat)."""

from .. import compat as _compat

_compat.install()
