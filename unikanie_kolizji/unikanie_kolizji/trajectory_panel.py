#!/usr/bin/env python3

"""Per-method trajectory store and a 2-D top-view matplotlib panel.

The ``TrajectoryStore`` holds per-CAS-method buffers that are filled by the
GUI as each ``/drone_i/gps`` message arrives.  The ``TrajectoryPanel``
displays all three methods side-by-side with their drone traces coloured
using the shared ``world_generator.color_hex()`` palette.
"""

import math

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFrame, QPushButton, QScrollArea, QSplitter,  # noqa: E402
                              QVBoxLayout, QWidget, QLabel, QHBoxLayout)

from . import option_compatibility as compat
from .world_generator import color_hex, target_positions, spawn_x

CAS_METHODS = list(compat.CAS_OPTIONS)   # ['3d_swap', 'gps', 'dpso']

# Fixed top-view window in metres.  The grid never rescales with the window
# size or the number of drones, so a trace can not become invisible just
# because a larger flock spread out; users scroll instead.
# 24 m x 18 m box: x in [-12, 12], y in [-5, 13].
GRID_X_MIN, GRID_X_MAX = -12.0, 12.0
GRID_Y_MIN, GRID_Y_MAX = -5.0, 13.0
CANVAS_PX = 500           # fixed canvas pixel size (=> scrollbars, no shrink)
CANVAS_H = 640            # taller than wide: bottom strip holds the vertical legend

# ---------- world-coordinate converter (matches drone_gps / avoidance) ----------
_RAD_LAT = 50.0
_COS_RAD = math.cos(math.radians(_RAD_LAT))


def navsat_to_xy(lat: float, lon: float):
    x = (lat - _RAD_LAT) * 111320.0
    y = (lon - 19.0) * 111320.0 * _COS_RAD
    return x, y


# ---------- trajectory data store -------------------------------------------------

class TrajectoryStore:
    """Append-only per-method, per-drone position buffer."""

    def __init__(self):
        self._data = {}

    def reset(self, method: str, drone_count: int, drone_names, targets):
        self._data[method] = {
            'n': drone_count,
            'names': list(drone_names),
            'targets': [(t[0], t[1]) for t in targets],
            'points': {name: [] for name in drone_names},
            'elapsed': None,
            'collisions': None,
        }

    def clear(self, method: str):
        self._data[method] = None

    def record(self, method: str, name: str, x: float, y: float, z: float):
        bucket = self._data.get(method)
        if bucket is None or name not in bucket['points']:
            return
        bucket['points'][name].append((x, y, z))

    def set_elapsed(self, method: str, seconds: float):
        bucket = self._data.get(method)
        if bucket is None:
            return
        bucket['elapsed'] = seconds

    def set_collisions(self, method: str, count: int):
        bucket = self._data.get(method)
        if bucket is None:
            return
        bucket['collisions'] = count

    def snapshot(self, method: str):
        return self._data.get(method)

    def traces(self, method: str):
        """Per-drone ``(xs, ys, zs)`` numpy arrays for ``method``.

        Also returned: the ``(targets)`` and the raw dict.  Used by the 3-D
        panel which styles each trace exactly like the 2-D one (same palette,
        same start/end markers, same fixed world limits) but projects it
        onto a third height axis instead of a top view.
        """
        data = self._data.get(method)
        if data is None:
            return None, None
        pts = data['points']
        out = {}
        for name, rows in pts.items():
            if not rows:
                out[name] = (np.empty(0), np.empty(0), np.empty(0))
                continue
            arr = np.asarray(rows)
            out[name] = (arr[:, 0], arr[:, 1], arr[:, 2])
        return out, data['targets']



# ---------- matplotlib 2-D panel ---------------------------------------------------

class TrajectoryPanel(QWidget):
    """Three side-by-side 2-D top-view figures, one per CAS method."""

    save_requested = pyqtSignal()

    def __init__(self, store: TrajectoryStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._axes = {}
        self._canvases = {}
        self._setup_ui()

    # --- UI ---------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('Trajectories (2-D top view)'))
        toolbar.addStretch()
        clear_all = QPushButton('Clear all')
        clear_all.clicked.connect(self._clear_all)
        toolbar.addWidget(clear_all)
        save_view = QPushButton('Save view')
        save_view.clicked.connect(self.save_requested.emit)
        toolbar.addWidget(save_view)
        outer.addLayout(toolbar)

        # Fixed-size canvases live inside a scroll area: the grid never
        # shrinks to fit the window or a bigger flock, it scrolls instead.
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        for m in CAS_METHODS:
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 4, 4, 4)

            # Fixed margins (no tight_layout): the axes box is a constant
            # fraction of the fixed canvas, so the grid square never changes
            # size even as legends/titles/tick labels vary.  The wide left
            # margin keeps the 'y (m)' axis label fully inside the figure.
            fig = Figure(figsize=(CANVAS_PX / 100.0, CANVAS_H / 100.0))
            fig.subplots_adjust(left=0.10, right=0.97, bottom=0.40, top=0.94)
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(CANVAS_PX, CANVAS_H)
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box', anchor='C')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            self._axes[m] = ax
            self._canvases[m] = canvas

            lay.addWidget(
                QLabel(m.replace('_', ' ').title()),
                alignment=Qt.AlignCenter)
            lay.addWidget(canvas, alignment=Qt.AlignCenter)

            btn = QPushButton('Clear')
            btn.clicked.connect(lambda _, mm=m: self._clear(mm))
            lay.addWidget(btn, alignment=Qt.AlignRight)
            splitter.addWidget(container)

        splitter.setSizes([CANVAS_PX] * len(CAS_METHODS))
        content_lay.addWidget(splitter)

        # Give the content a real minimum so the scroll area shows scrollbars
        # instead of squeezing the canvases when the window is too small.
        content.setMinimumSize(
            CANVAS_PX * len(CAS_METHODS) + 40,
            CANVAS_H + 90)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

    # --- refresh ----------------------------------------------------------------

    def refresh(self, method: str = None):
        methods = [method] if method else CAS_METHODS
        for m in methods:
            self._redraw(m)

    def _redraw(self, m: str):
        ax = self._axes[m]
        ax.clear()
        ax.set_aspect('equal', adjustable='box', anchor='C')
        ax.set_xlim(GRID_X_MIN, GRID_X_MAX)
        ax.set_ylim(GRID_Y_MIN, GRID_Y_MAX)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(m.replace('_', ' ').title())

        data = self.store.snapshot(m)
        if data is None:
            ax.text(0.5, 0.5, 'No run yet', transform=ax.transAxes,
                    ha='center', va='center', color='#888', fontsize=12)
            self._canvases[m].draw_idle()
            return

        for i, name in enumerate(data['names']):
            pts = data['points'][name]
            if not pts:
                continue
            xs, ys = zip(*((p[0], p[1]) for p in pts))
            c = color_hex(i)
            ax.plot(xs, ys, color=c, linewidth=1.5, label=name, zorder=2)
            ax.plot(xs[0], ys[0], marker='o', color=c, markersize=7,
                    markeredgecolor='k', zorder=3)          # start
            ax.plot(xs[-1], ys[-1], marker='o', color=c, markersize=9,
                    markeredgecolor='k', zorder=3)          # current

        for i, (tx, ty) in enumerate(data['targets']):
            ax.plot(tx, ty, marker='X', color=color_hex(i),
                    markersize=11, markeredgecolor='k', zorder=3)

        ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.10),
                  ncol=1, fontsize='small', frameon=True, handlelength=1.2)
        self._canvases[m].draw_idle()

        # Elapsed time text underneath the grid, immediately right of the
        # legend, with the collision count drawn one line beneath it.  Needs a
        # rendered legend to measure its extent, so draw once synchronously
        # before placing the text.
        self._canvases[m].draw()
        leg = ax.get_legend()
        elapsed = data.get('elapsed')
        label = (f'Total time: {elapsed:.2f} s'
                 if elapsed is not None else 'Total time: -- s')
        if leg is not None:
            lb = leg.get_window_extent()
            x0, y0 = ax.transAxes.inverted().transform((lb.x1 + 12, lb.y0))
        else:
            x0, y0 = 0.0, -0.10
        elapsed_text = ax.text(x0, y0, label, transform=ax.transAxes,
                               ha='left', va='top', fontsize='small',
                               color='#333')

        collisions = data.get('collisions')
        if collisions is not None:
            eb = elapsed_text.get_window_extent()
            cx, cy = ax.transAxes.inverted().transform(
                (eb.x0, eb.y0 - 2.0))
            ax.text(cx, cy, f'Collisions: {collisions}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize='small', color='#333')
        self._canvases[m].draw_idle()

    def _clear(self, m: str):
        self.store.clear(m)
        self._redraw(m)

    def _clear_all(self):
        for m in CAS_METHODS:
            self._clear(m)

    def figure(self, method: str):
        """The matplotlib Figure drawn for ``method`` (None if not built)."""
        canvas = self._canvases.get(method)
        return canvas.figure if canvas is not None else None
