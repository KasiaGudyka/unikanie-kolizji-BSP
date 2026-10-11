#!/usr/bin/env python3

"""Qt configuration + trajectory window for the drone simulation.

Main features:

  1. physics engine  - switching between engines usable for UAV Gazebo Harmonic
                       simulation (ode / tpe / bullet / dart)
  2. drone size      - drones of 0.5 kg (Parrot Bebop 2), 5 kg and 20 kg
3. strong wind     - off / constant 1, 3, 5, 10 m/s cross-wind / the same
                        speeds with a sinusoidal gust
                        (WindEffects field + native gz_aero_drag per-mass drag;
                        flight envelope: 1-3 m/s fine for every drone, 5 m/s is
                        held by every airframe at a small steady downwind
                        offset, gust modes add excursions; 10 m/s exceeds the
                        command envelope - documented limit)
  4. drone count     - number of drones to spawn (2 / 5 / 10)
  5. CAS method      - collision avoidance method (GPS / 3D-SWAP / DPSO /
                       convex hull / GMM / combined)
  6. run statistics  - optional saving of a finished-trajectory PNG plus the
                       measured time and collision count into a per-configuration
                       CSV table (stats/ directory)

  + A "Trajectories" tab that draws 2-D top-view traces of every drone for
    each CAS method side-by-side.  Launching a new sim overrides only the
    panel for the selected method; others persist for comparison.  A
    "Save view" button snapshots the current figure on demand.
"""

import os
import shutil
import signal
import subprocess
import sys
import time

import yaml
import rclpy
from rclpy.node import Node as _ROSNode
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float64, Int32
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
                              QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                              QMessageBox, QPushButton, QScrollArea,
                              QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
                              QTabWidget)

from . import option_compatibility as compat
from .trajectory_3d_panel import TrajectoryPanel3D as _Trajectory3DPanel
from . import stats_save
from .drone_gps import (ELAPSED_TOPIC, GO_TOPIC, VIEW_READY_TOPIC)
from .drone_start_gate import COLLISIONS_TOPIC
from .trajectory_panel import (TrajectoryStore, TrajectoryPanel,
                                navsat_to_xy, CAS_METHODS)
from .world_generator import spawn_x, target_positions


class NoWheelComboBox(QComboBox):
    """QComboBox subclass that ignores mouse-wheel events to prevent
    accidental selection changes while scrolling the window content."""

    def wheelEvent(self, event):
        event.ignore()

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PKG_DIR, 'config', 'sim_config.yaml')

ENGINE_LABELS = {
    'ode': 'ODE',
    'tpe': 'TPE',
    'bullet': 'Bullet',
    'dart': 'DART',
}
MASS_LABELS = {
    0.5: '0.5 kg (Parrot Bebop 2)',
    5.0: '5 kg',
    20.0: '20 kg',
}
CAS_LABELS = {
    'gps': 'GPS (no-fly-zone repulsion)',
    '3d_swap': '3D-SWAP',
    'dpso': 'DPSO (particle-swarm planner)',
    'convex': 'Convex hull',
    'gmm': 'Gaussian mixture (GMM)',
    'combined': 'Combined (DPSO+GMM+hull)',
}


class DroneConfigGUI(QMainWindow):
    """Main configuration + trajectory window for the drone simulation."""

    def __init__(self, output_path: str = DEFAULT_OUTPUT,
                 stats_dir: str = None):
        super().__init__()
        self.output_path = output_path
        self._stats_dir = stats_dir or stats_save.DEFAULT_STATS_DIR
        self.setWindowTitle('Drone Simulation Configuration')
        self.resize(800, 720)
        self.setMinimumSize(480, 400)

        # Trajectory data store + panel.
        self._store = TrajectoryStore()
        self._panel = TrajectoryPanel(self._store)
        self._panel.save_requested.connect(self._save_manual_snapshot)
        self._panel_3d = _Trajectory3DPanel(self._store, 'gps')
        self._panel_3d.save_requested.connect(self._save_manual_snapshot_3d)

        # ROS state (initialised lazily on first launch).
        self._ros_node = None
        self._subs = []
        self._go_persist_sub = None
        self._view_ready_pub = None
        self._view_ready_published = False
        self._active_method = None
        self._pending_refresh = False
        self._run_config = None
        self._go_received = False

        self.setup_ui()
        self._fit_width_to_content()
        self.refresh_compatibility()

    def _fit_width_to_content(self):
        """Open wide enough that the trajectory group never needs enlarging.

        The long (non-wrapping) checkbox labels in section 6 define the widest
        content in the config tab; without this the vertical scroll area
        grows a horizontal scrollbar and the window must be widened by hand.
        """
        width = 800
        for name in ('show_traj_check', 'save_stats_check'):
            widget = getattr(self, name, None)
            if widget is not None:
                width = max(width, widget.sizeHint().width())
        if width > 800:
            # Config-tab container margins (2x12), group-box insets (~2x9),
            # tab/scroll chrome and the vertical scrollbar, counted once.
            self.resize(width + 72, 720)

    # --- UI construction ----------------------------------------------------

    def setup_ui(self):
        config_tab = self._build_config_tab()

        tabs = QTabWidget()
        tabs.addTab(config_tab, 'Configuration')
        tabs.addTab(self._panel, 'Trajectories')
        self.tabs = tabs
        self.setCentralWidget(tabs)

    def _build_config_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        scroll.setWidget(container)
        root = QVBoxLayout(container)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel('Drone Simulation Configuration')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        root.addWidget(title)

        subtitle = QLabel(
            'Options configure the ongoing simulation. Selections are '
            'validated against a safety compatibility matrix in real time.')
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

        # ---- Engine selector (feature 1) -----------------------------------
        engine_box = QGroupBox('1. Physics engine (Gazebo Harmonic)')
        engine_form = QFormLayout(engine_box)
        self.engine_combo = NoWheelComboBox()
        for eng in compat.ENGINE_OPTIONS:
            self.engine_combo.addItem(ENGINE_LABELS[eng], eng)
            self.engine_combo.model().item(self.engine_combo.count() - 1, 0).setEnabled(
                eng in compat.allowed_engines(0.5))
        self.engine_combo.currentIndexChanged.connect(self.refresh_compatibility)
        engine_form.addRow('Engine:', self.engine_combo)
        self.make_form_field_flexible(engine_form, self.engine_combo)
        self.engine_note = QLabel('')
        self.engine_note.setWordWrap(True)
        self.engine_note.setStyleSheet('color: #9e9e9e; font-size: 11px;')
        engine_form.addRow(self.engine_note)
        root.addWidget(engine_box)

        # ---- Drone size selector (feature 2) -------------------------------
        size_box = QGroupBox('2. Drone size')
        size_form = QFormLayout(size_box)
        self.size_combo = NoWheelComboBox()
        for mass in compat.DRONE_MASS_OPTIONS:
            self.size_combo.addItem(MASS_LABELS[mass], mass)
        self.size_combo.currentIndexChanged.connect(self.on_size_changed)
        size_form.addRow('Mass:', self.size_combo)
        self.make_form_field_flexible(size_form, self.size_combo)
        self.size_note = QLabel('')
        self.size_note.setWordWrap(True)
        self.size_note.setStyleSheet('color: #9e9e9e; font-size: 11px;')
        size_form.addRow(self.size_note)
        root.addWidget(size_box)

        # ---- Wind selector (feature 3) -------------------------------------
        wind_box = QGroupBox('3. Strong wind')
        wind_form = QFormLayout(wind_box)
        self.wind_combo = NoWheelComboBox()
        self.wind_combo.addItem('Off', 'off')
        for speed in (1, 3, 5, 10):
            self.wind_combo.addItem(f'Constant {speed} m/s', f'strong_{speed}')
            self.wind_combo.addItem(f'Gust {speed} m/s', f'gust_{speed}')
        self.wind_combo.currentIndexChanged.connect(self.refresh_compatibility)
        wind_form.addRow('Wind:', self.wind_combo)
        self.make_form_field_flexible(wind_form, self.wind_combo)
        self.wind_note = QLabel('')
        self.wind_note.setWordWrap(True)
        self.wind_note.setStyleSheet('color: #9e9e9e; font-size: 11px;')
        wind_form.addRow(self.wind_note)
        root.addWidget(wind_box)

        # ---- Drone count selector (feature 4) ------------------------------
        count_box = QGroupBox('4. Number of drones')
        count_form = QFormLayout(count_box)
        self.count_combo = NoWheelComboBox()
        for n in compat.DRONE_COUNT_OPTIONS:
            self.count_combo.addItem(str(n), n)
        self.count_combo.currentIndexChanged.connect(self.refresh_compatibility)
        count_form.addRow('Drones:', self.count_combo)
        self.make_form_field_flexible(count_form, self.count_combo)
        root.addWidget(count_box)

        # ---- CAS selector (feature 5) --------------------------------------
        cas_box = QGroupBox('5. Collision avoidance method (CAS)')
        cas_form = QFormLayout(cas_box)
        self.cas_combo = NoWheelComboBox()
        for method in compat.CAS_OPTIONS:
            self.cas_combo.addItem(CAS_LABELS[method], method)
        self.cas_combo.currentIndexChanged.connect(self.refresh_compatibility)
        cas_form.addRow('Method:', self.cas_combo)
        self.make_form_field_flexible(cas_form, self.cas_combo)
        self.cas_note = QLabel('')
        self.cas_note.setWordWrap(True)
        self.cas_note.setStyleSheet('color: #9e9e9e; font-size: 11px;')
        cas_form.addRow(self.cas_note)
        root.addWidget(cas_box)

        # ---- Trajectory visualization toggle --------------------------------
        traj_box = QGroupBox('6. Trajectory visualization')
        traj_layout = QVBoxLayout(traj_box)
        self.show_traj_check = QCheckBox(
            'Show trajectories - record GPS tracks during this run and '
            'display them on the Trajectories tab.')
        self.show_traj_check.setChecked(True)
        self.show_traj_check.toggled.connect(self.refresh_compatibility)
        traj_layout.addWidget(self.show_traj_check)
        self.save_stats_check = QCheckBox(
            'Save run data - after each run store a PNG of the finished '
            'trajectory and append its measured time to the per-config '
            'statistics CSV in the stats/ directory.')
        traj_layout.addWidget(self.save_stats_check)
        self.record_check = QCheckBox(
            'Record run data - write the .tlog telemetry log and the ros2 bag '
            'for this run (toggle off to compare load time with/without '
            'recording).')
        self.record_check.setChecked(True)
        traj_layout.addWidget(self.record_check)
        root.addWidget(traj_box)

        # ---- Status / validation panel -------------------------------------
        status_box = QGroupBox('Safety & compatibility check')
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_layout.addWidget(self.status_label)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(100)
        # Most of the extra vertical space goes to the summary area.
        self.summary_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        status_layout.addWidget(self.summary_text)
        root.addWidget(status_box)

        # ---- Action buttons ------------------------------------------------
        buttons = QHBoxLayout()
        self.launch_btn = QPushButton('Launch Simulation')
        self.launch_btn.clicked.connect(self.on_launch)
        quit_btn = QPushButton('Quit')
        quit_btn.clicked.connect(self.close)
        buttons.addWidget(self.launch_btn)
        buttons.addWidget(quit_btn)
        root.addLayout(buttons)

        self.footer = QLabel(f'Config output: {self.output_path}\n'
                         f'Stats output: {self._stats_dir}')
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet('color: #888888; font-size: 11px;')
        root.addWidget(self.footer)
        return scroll

    @staticmethod
    def make_form_field_flexible(form, field):
        """Labels stay stiff; the field widget stretches with the window.

        QFormLayout allocates spare space to fields that may grow, so the
        combo boxes (and any note labels) flex while the row labels keep
        their natural width as the window is resized.
        """
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for i in range(form.rowCount()):
            label_item = form.itemAt(i, QFormLayout.LabelRole)
            if label_item is not None and label_item.widget() is not None:
                label_item.widget().setSizePolicy(QSizePolicy.Fixed,
                                                  QSizePolicy.Preferred)

    # --- ROS helpers --------------------------------------------------------

    def _ensure_ros(self):
        if self._ros_node is not None:
            return
        if not rclpy.ok():
            rclpy.init()
        self._ros_node = rclpy.create_node('trajectory_monitor')
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(
            lambda: rclpy.spin_once(self._ros_node, timeout_sec=0.0)
            if rclpy.ok() else None)
        self._spin_timer.start(10)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._flush_refresh)
        self._refresh_timer.start(100)
        # Latched one-shot view_ready pub — signalled the very first time the
        # maneuver-go subscription fires, so the gate's transient sub sees it.
        if self._view_ready_pub is None:
            self._view_ready_pub = self._ros_node.create_publisher(
                Bool, VIEW_READY_TOPIC,
                QoSProfile(depth=10,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Persistent (TRANSIENT_LOCAL) go sub — one for the process lifetime,
        # never recreated per run, so the gui can't spawn a reset/ghost go.
        self._go_persist_sub = self._ros_node.create_subscription(
            Bool, GO_TOPIC, self._go_cb,
            QoSProfile(depth=10,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Persistent per-prefix GPS subs (diff-created per run in on_launch).
        self._gps_subs = {}

    def _go_cb(self, msg: Bool):
        """Latch the maneuver-go so recording gating is stable across ticks."""
        if self._active_method is None:
            return
        if msg.data:
            self._go_received = True
            self._pending_refresh = True

    def _gps_cb(self, name: str, msg: NavSatFix):
        if self._active_method is None or not self._go_received:
            return
        x, y = navsat_to_xy(msg.latitude, msg.longitude)
        self._store.record(self._active_method, name, x, y, msg.altitude)
        self._pending_refresh = True

    def _elapsed_cb(self, msg: Float64):
        if self._active_method is None:
            return
        self._store.set_elapsed(self._active_method, msg.data)
        self._panel.refresh(self._active_method)
        self._pending_refresh = False
        if self.save_stats_check.isChecked():
            self._save_run_data()

    def _collisions_cb(self, msg: Int32):
        if self._active_method is None:
            return
        self._store.set_collisions(self._active_method, msg.data)
        self._panel.refresh(self._active_method)

    def _save_run_data(self):
        cfg = self._run_config
        fig = self._panel.figure(self._active_method)
        if cfg is None or self._active_method is None or fig is None:
            return
        snap = self._store.snapshot(self._active_method)
        elapsed = snap['elapsed'] if snap is not None else None
        collisions = snap['collisions'] if snap is not None else None
        try:
            stats_save.save_run(self._stats_dir, cfg['cas'],
                                cfg['drone_count'], cfg['drone_mass'],
                                cfg['wind'], elapsed, fig,
                                collisions=collisions)
            print(f'[stats] saved run data to {self._stats_dir}')
        except Exception as exc:  # noqa: BLE001
            print(f'[stats] save failed: {exc}', file=sys.stderr)

    def _save_manual_snapshot(self):
        cfg = self._run_config
        fig = self._panel.figure(self._active_method)
        if cfg is None or self._active_method is None or fig is None:
            return
        try:
            path = stats_save.save_png_snapshot(
                self._stats_dir, cfg['cas'], cfg['drone_count'],
                cfg['drone_mass'], cfg['wind'], fig)
            print(f'[stats] saved view: {path}')
        except Exception as exc:  # noqa: BLE001
            print(f'[stats] snapshot failed: {exc}', file=sys.stderr)

    def _save_manual_snapshot_3d(self):
        """Manual save of the 3-D view (no CSV row is appended)."""
        cfg = self._run_config
        fig = self._panel_3d.figure()
        if cfg is None or self._active_method is None or fig is None:
            return
        try:
            path = stats_save.save_png_snapshot(
                self._stats_dir, cfg['cas'], cfg['drone_count'],
                cfg['drone_mass'], cfg['wind'], fig)
            print(f'[stats] saved 3-D view: {path}')
        except Exception as exc:  # noqa: BLE001
            print(f'[stats] 3-D snapshot failed: {exc}', file=sys.stderr)

    def _pending_merge_cleanup(self):
        pass

    def _flush_refresh(self):
        if self._active_method is None:
            return
        # view_ready is signalled once per run, independent of the panel
        # redraw gate, so the start gate is released without waiting on data
        # to arrive (which would otherwise stall the maneuver for the full
        # view_ready_timeout).
        if not self._view_ready_published \
                and self._view_ready_pub is not None:
            self._view_ready_pub.publish(Bool(data=True))
            self._view_ready_published = True
        if not self._pending_refresh:
            return
        self._pending_refresh = False
        self._panel.refresh(self._active_method)
    # --- Selection handling -------------------------------------------------

    def selected(self) -> dict:
        """Return the current raw selection."""
        return {
            'drone_mass': self.size_combo.currentData(),
            'engine': self.engine_combo.currentData(),
            'wind': self.wind_combo.currentData(),
            'drone_count': self.count_combo.currentData(),
            'cas': self.cas_combo.currentData(),
        }

    def on_size_changed(self):
        # Safety self-correction: restrict the engine list to what the chosen
        # size allows, resetting an invalid engine to the first allowed one.
        mass = self.size_combo.currentData()
        allowed = compat.allowed_engines(mass)

        for i in range(self.engine_combo.count()):
            eng = self.engine_combo.itemData(i)
            item = self.engine_combo.model().item(i, 0)
            item.setEnabled(eng in allowed)

        if self.engine_combo.currentData() not in allowed and allowed:
            allowed_index = [
                self.engine_combo.itemData(i)
                for i in range(self.engine_combo.count())].index(allowed[0])
            self.engine_combo.setCurrentIndex(allowed_index)

        if mass == 0.5:
            size_note = ('0.5 kg Parrot Bebop 2 model: only the ODE engine is '
                         'supported by the multicopter control system.')
        elif mass in (5.0, 20.0):
            from unikanie_kolizji import world_generator as _wg
            spec = _wg.AIRFRAME_SPECS[mass]
            wheelbase_mm = round(2.0 * spec['arm_m'] * 1000.0)
            rpm = int(round(_wg.hover_rpm(spec)))
            size_note = (
                f'Real {spec["name"]}: flight mass {spec["mass_kg"]:g} kg, '
                f'rotor arm {spec["arm_m"]:.3f} m (wheelbase {wheelbase_mm} '
                f'mm), rotor thrust constant {spec["rotor_k"]:g}, real hover '
                f'~{rpm} RPM, motor cap {spec["max_rot_velocity"]} rad/s. '
                'The flight controller is deliberately detuned so heavier '
                'drones fly sedately and stably, like a commercial cargo UAV.')
        else:
            size_note = ''
        self.size_note.setText(size_note)
        self.refresh_compatibility()

    def refresh_compatibility(self):
        sel = self.selected()
        result = compat.validate(sel['drone_mass'], sel['engine'], sel['wind'],
                                 sel['drone_count'], sel['cas'])
        if result.valid:
            self.status_label.setStyleSheet(
                'color: #2e7d32; font-weight: bold;')
            self.status_label.setText('OK - selection is compatible.')
        else:
            self.status_label.setStyleSheet(
                'color: #c62828; font-weight: bold;')
            self.status_label.setText('BLOCKED - selection is not compatible.')

        lines = []
        for e in result.errors:
            lines.append(f'[ERROR] {e}')
        for w in result.warnings:
            lines.append(f'[WARNING] {w}')
        if not lines:
            lines.append('No issues.')
        resolved = compat.resolve(**sel)
        lines.append(f'Resolved engine: {resolved["engine"].upper()}')
        self.summary_text.setPlainText('\n'.join(lines))

        self.wind_note.setText(
            '' if sel['wind'] == 'off' else
            f'Wind {compat.WIND_SPEED_MS[sel["wind"]]:g} m/s cross-wind'
            + (' with a sinusoidal gust.' if sel['wind'].startswith('gust') else '.'))
        self.engine_note.setText(
            '' if sel['engine'] in compat.allowed_engines(sel['drone_mass'])
            else 'Engine incompatible with the selected drone size.')

    # --- Output -------------------------------------------------------------

    def selection_dict(self) -> dict:
        sel = self.selected()
        res = compat.resolve(**sel)
        return {
            'physics_engine': res['engine'],
            'drone_mass_kg': res['drone_mass'],
            'wind': res['wind'],
            'wind_speed_ms': compat.WIND_SPEED_MS[res['wind']],
            'drone_count': res['drone_count'],
            'cas_method': compat.cas_launch_value(res['cas']),
            'show_trajectories': self.show_traj_check.isChecked(),
            'recording_enabled': self.record_check.isChecked(),
            'compatibility': {
                'valid': res['validation'].valid,
                'errors': res['validation'].errors,
                'warnings': res['validation'].warnings,
            },
        }

    def launch_command(self) -> str:
        sel = self.selection_dict()
        return (
            'ros2 launch unikanie_kolizji gazebo_two_drones.launch.py '
            f'physics_engine:={sel["physics_engine"]} '
            f'drone_count:={sel["drone_count"]} '
            f'drone_size_kg:={sel["drone_mass_kg"]} '
            f'wind:={sel["wind"]} '
            f'collision_avoidance_method:={sel["cas_method"]} '
            f'recording_enabled:='
            f'{"true" if sel["recording_enabled"] else "false"}')

    def on_launch(self):
        sel = self.selected()
        result = compat.validate(sel['drone_mass'], sel['engine'], sel['wind'],
                                 sel['drone_count'], sel['cas'])
        if not result.valid:
            QMessageBox.critical(
                self, 'Incompatible selection',
                'The current selection is not compatible:\n\n' +
                '\n'.join(result.errors))
            return

        # --- trajectory capture setup ---
        # Reset the one-shot view_ready handshake so every run re-arms the
        # start gate on the next refresh tick (or immediately below when the
        # trajectory panels are disabled).
        self._view_ready_published = False
        if not self.show_traj_check.isChecked():
            self._active_method = None
            self._pending_refresh = False
            self._run_config = None
            for s in self._subs:
                s.destroy()
            self._subs.clear()
            self._ensure_ros()
            if self._view_ready_pub is not None:
                self._view_ready_pub.publish(Bool(data=True))
                self._view_ready_published = True
        else:
            n = sel['drone_count']
            names = [f'drone_{i}' for i in range(n)]
            tgt = target_positions(spawn_x(n))
            self._ensure_ros()
            # Persistent per-prefix GPS subs: only diff-create, so run-to-run
            # transitions never destroy/recreate a subscription mid-run.
            old = set(self._gps_subs)
            new = set(names)
            for prefix in old - new:
                self._gps_subs.pop(prefix).destroy()
            for prefix in new - old:
                self._gps_subs[prefix] = self._ros_node.create_subscription(
                    NavSatFix, f'/{prefix}/gps',
                    (lambda nm: lambda msg: self._gps_cb(nm, msg))(prefix),
                    QoSProfile(depth=10,
                               durability=DurabilityPolicy.VOLATILE))
            for s in self._subs:
                s.destroy()
            self._subs.clear()
            self._store.reset(sel['cas'], n, names, tgt)
            self._active_method = sel['cas']
            self._pending_refresh = False
            self._go_received = False
            self._run_config = dict(sel)
            # Maneuver elapsed time (volatile topic, published once when every
            # drone reaches its target / the goal timeout elapses) and the
            # aggregate collision count (published live during the maneuver).
            # (The maneuver-go is delivered by the persistent _go_persist_sub,
            # never recreated per run, so a stale latched go can't replay.)
            self._subs.append(
                self._ros_node.create_subscription(
                    Float64, ELAPSED_TOPIC, self._elapsed_cb, 10))
            self._subs.append(
                self._ros_node.create_subscription(
                    Int32, COLLISIONS_TOPIC, self._collisions_cb, 10))

        config = self.selection_dict()
        config['launch_command'] = self.launch_command()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        self.apply_to_simulation(config)

        print(f'\n[sim_gui] Validated configuration written to:\n'
              f'{self.output_path}\n\n'
              f'Launching:\n{config["launch_command"]}\n\n'
              'Simulation output appears in the terminal that started the GUI.')

        if self.show_traj_check.isChecked():
            self.tabs.setCurrentIndex(1)

    def _kill_stray_stack(self):
        """Kill leftover sim stack processes by name, whatever their parent.

        A crashed or SIGKILLed ``ros2 launch`` parent leaves its children (the
        per-drone nodes, the gate with its bag recorder, the bridge, gz sim)
        reparented to systemd/init, where the ``sim_proc`` process-group
        teardown can never reach them.  Sweep by stable name pattern instead so
        no stray from a previous run can spin into the next one.  Uses
        bracketed regexes so pkill never matches this GUI's own cmdline.
        """
        patterns = [
            'd[r]one_gps',
            'd[r]one_start_gate',
            'd[r]one_collision_avoidance',
            'd[r]one_labels',
            'r[o]s2 [b]ag record',
            'ros_gz_bridge/[p]arameter_bridge',
            'gz-[s]im-server',
            'gz [s]im',
            'ros2 [l]aunch.*gazebo_two_drones',
        ]
        # Each pkill must complete before the new launch is spawned, or it
        # would race the fresh "ros2 launch ... gazebo_two_drones" cmdline and
        # kill it (run() blocks until pkill has finished scanning /proc).
        for pat in patterns:
            try:
                subprocess.run(['pkill', '-TERM', '-f', pat],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            except Exception:  # noqa: BLE001
                continue
        time.sleep(2.0)
        for pat in patterns:
            try:
                subprocess.run(['pkill', '-KILL', '-f', pat],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            except Exception:  # noqa: BLE001
                continue
        time.sleep(1.0)

    def apply_to_simulation(self, config: dict):
        """Launch the real stack selected in the GUI.

        The N-drone world is generated at launch time (world_generator), the
        selected engine/mass/wind/count/CAS are forwarded as launch arguments,
        and gz runs with a transient ``-z`` seed. The child process inherits
        this terminal so its logs stream to stdout.
        """
        cmd = self.launch_command()
        print(f'[sim_gui] launching: {cmd}')
        if shutil.which('ros2') is None:
            QMessageBox.warning(
                self, 'ros2 not found',
                'The ros2 CLI was not found on PATH. Source your ROS 2 setup '
                'before running the GUI.')
            return
        # Tear down any previous run first: its transient-latched ready/go/
        # goal_reached publishers would otherwise replay into the fresh gate,
        # forcing a ~0s (or ghost) maneuver on the next run.
        self._kill_stray_stack()
        old = getattr(self, 'sim_proc', None)
        if old is not None and old.poll() is None:
            print(f'[sim_gui] terminating previous launch (PID {old.pid}).')
            try:
                os.killpg(os.getpgid(old.pid), signal.SIGTERM)
            except (OSError, PermissionError):
                old.terminate()
            try:
                old.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(old.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.sim_proc = subprocess.Popen(
                cmd.split(), start_new_session=True)
        except OSError as e:
            QMessageBox.critical(
                self, 'Launch failed',
                f'Could not start "ros2 launch":\n{e}')
            return
        print(f'[sim_gui] launch started (PID {self.sim_proc.pid}).')

    def closeEvent(self, event):
        proc = getattr(self, 'sim_proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        self._kill_stray_stack()
        for t in (getattr(self, '_spin_timer', None),
                  getattr(self, '_refresh_timer', None)):
            if t is not None:
                t.stop()
        node = getattr(self, '_ros_node', None)
        if node is not None and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            try:
                rclpy.try_shutdown()
            except Exception:  # noqa: BLE001
                pass
        event.accept()


def main(args=None):
    app = QApplication(sys.argv[:1])
    gui = DroneConfigGUI()
    gui.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
