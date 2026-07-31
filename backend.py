"""
Matplotlib backend selection.

Importing matplotlib.pyplot binds a backend for the whole process, and forcing
an interactive one at import time makes every module unimportable on a machine
without a display -- which includes CI and any headless box. Backend choice is
therefore deferred until something actually wants to draw, and falls back to the
non-interactive Agg writer when no GUI toolkit is available.

Set the MPLBACKEND environment variable to override the choice entirely.
"""

import os

import matplotlib

#: Interactive backends tried in order when a window is requested.
INTERACTIVE_BACKENDS = ('TkAgg', 'QtAgg', 'MacOSX')


def select_backend(interactive: bool = True) -> str:
    """Bind a matplotlib backend and return its name.

    Args:
        interactive: True when the caller intends to open a window. False
            selects Agg directly, which is what saving files wants.

    Returns:
        The name of the backend that ended up active.
    """
    if os.environ.get('MPLBACKEND'):
        # Respect an explicit user choice without second-guessing it.
        return matplotlib.get_backend()

    if not interactive:
        matplotlib.use('Agg', force=True)
        return matplotlib.get_backend()

    for candidate in INTERACTIVE_BACKENDS:
        try:
            matplotlib.use(candidate, force=True)
            return matplotlib.get_backend()
        except Exception:
            continue

    matplotlib.use('Agg', force=True)
    return matplotlib.get_backend()


def has_display() -> bool:
    """True when an interactive backend is currently active"""
    return matplotlib.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'template')
