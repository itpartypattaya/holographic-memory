"""The holographic engine, vendored from itpartypattaya/hermes-plugin-holographic v0.25.1 (cd8848c28529c37256509347e462764fd6372af2) by tools/sync_engine.py.

Do not edit the modules here: change the plugin and vendor its next tag. This file is the only one written by the
sync; it supplies the Hermes core helpers the store calls when Hermes core is not installed (holomem.compat)."""

from .. import compat as _compat

_compat.install()
