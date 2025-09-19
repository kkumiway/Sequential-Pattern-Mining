# gui_prefixspan.py
# PyQt6 GUI wrapper for your PrefixSpan implementation

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Import your algorithm(s)
from algo.prefixspan import PrefixSpan


@dataclass
class RunConfig:
    algorithm_key: str
    input_csv: str
    output_file: str
    minsup_relative: float
    max_pattern_length: int
    min_len: int


class MinerWorker(QThread):
    """Background worker that runs the selected algorithm to avoid blocking the UI."""

    started_log = pyqtSignal(str)
    finished_log = pyqtSignal(str)
    progress_log = pyqtSignal(str)
    error = pyqtSignal(str)
    done = pyqtSignal(int, int)  # total_time_ms, pattern_count

    def __init__(self, cfg: RunConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self) -> None:
        try:
            self.started_log.emit("Starting mining...\n")
            algo_key = self.cfg.algorithm_key

            # Algorithm registry (easily extensible)
            registry: Dict[str, Callable[[RunConfig], tuple[int, int]]] = {
                "prefixspan": self._run_prefixspan,
            }

            if algo_key not in registry:
                raise ValueError(f"Unknown algorithm: {algo_key}")

            total_ms, count = registry[algo_key](self.cfg)
            self.finished_log.emit("Mining finished.\n")
            self.done.emit(total_ms, count)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Error: {e}\n\n{tb}")

    # --- Individual runners ---
    def _run_prefixspan(self, cfg: RunConfig) -> tuple[int, int]:
        algo = PrefixSpan(maximum_pattern_length=cfg.max_pattern_length, min_len=cfg.min_len)
        self.progress_log.emit(
            f"Running PrefixSpan on '{cfg.input_csv}' -> '{cfg.output_file}'\n"
        )
        self.progress_log.emit(
            f"minsup (rel) = {cfg.minsup_relative}, max_len = {cfg.max_pattern_length}, min_len = {cfg.min_len}\n"
        )
        algo.run(cfg.input_csv, cfg.minsup_relative, cfg.output_file)
        return algo.total_time_ms, algo.pattern_count


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Pattern Mining GUI")
        self.setMinimumSize(760, 560)

        # --- Widgets ---
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["prefixspan"])  # extend later

        self.le_input = QLineEdit()
        self.le_output = QLineEdit()
        self.btn_browse_input = QPushButton("Browse…")
        self.btn_browse_output = QPushButton("Save as…")

        self.spin_minsup = QDoubleSpinBox()
        self.spin_minsup.setRange(0.0, 1.0)
        self.spin_minsup.setSingleStep(0.05)
        self.spin_minsup.setDecimals(3)
        self.spin_minsup.setValue(0.5)
        self.spin_minsup.setToolTip("Relative minimum support (0–1)")

        self.spin_maxlen = QSpinBox()
        self.spin_maxlen.setRange(1, 100000)
        self.spin_maxlen.setValue(10)
        self.spin_maxlen.setToolTip("Maximum pattern length")

        self.spin_minlen = QSpinBox()
        self.spin_minlen.setRange(1, 100000)
        self.spin_minlen.setValue(1)  # default: behave like original
        self.spin_minlen.setToolTip("Minimum pattern length to WRITE (by item count)")

        self.btn_run = QPushButton("Run")
        self.btn_run.setDefault(True)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)  # will switch to busy state (0,0) while running
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setStyleSheet("color: #666;")

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Logs will appear here…")

        self.lbl_elapsed = QLabel("Elapsed: 0 ms")
        self.lbl_count = QLabel("Patterns: 0")

        # --- Layout ---
        form = QFormLayout()
        form.addRow("Algorithm", self.combo_algo)

        in_row = QHBoxLayout()
        in_row.addWidget(self.le_input, 1)
        in_row.addWidget(self.btn_browse_input)
        form.addRow("Input CSV", in_row)

        out_row = QHBoxLayout()
        out_row.addWidget(self.le_output, 1)
        out_row.addWidget(self.btn_browse_output)
        form.addRow("Output file", out_row)

        form.addRow("minsup (rel)", self.spin_minsup)
        form.addRow("Max pattern length", self.spin_maxlen)
        form.addRow("Min pattern length", self.spin_minlen)  

        params_box = QGroupBox("Run Configuration")
        params_box.setLayout(form)

        run_row = QHBoxLayout()
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.progress, 1)
        run_row.addWidget(self.lbl_status)

        stats_row = QHBoxLayout()
        stats_row.addWidget(self.lbl_elapsed)
        stats_row.addSpacing(20)
        stats_row.addWidget(self.lbl_count)
        stats_row.addStretch(1)

        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(params_box)
        v.addLayout(run_row)
        v.addLayout(stats_row)
        v.addWidget(self.txt_log, 1)
        self.setCentralWidget(central)

        # --- Signals ---
        self.btn_browse_input.clicked.connect(self.choose_input)
        self.btn_browse_output.clicked.connect(self.choose_output)
        self.btn_run.clicked.connect(self.start_run)

        self.worker: Optional[MinerWorker] = None

    # --- UI helpers ---
    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose input CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.le_input.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save output as", "patterns.txt", "Text files (*.txt);;All files (*)")
        if path:
            self.le_output.setText(path)

    def start_run(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return
        input_csv = self.le_input.text().strip()
        output_file = self.le_output.text().strip()
        if not input_csv:
            QMessageBox.warning(self, "Missing input", "Please choose an input CSV file.")
            return
        if not output_file:
            QMessageBox.warning(self, "Missing output", "Please choose an output file.")
            return

        cfg = RunConfig(
            algorithm_key=self.combo_algo.currentText(),
            input_csv=input_csv,
            output_file=output_file,
            minsup_relative=float(self.spin_minsup.value()),
            max_pattern_length=int(self.spin_maxlen.value()),
            min_len=int(self.spin_minlen.value()),
        )

        self.txt_log.clear()
        self.set_running(True)

        self.worker = MinerWorker(cfg)
        self.worker.started_log.connect(self.append_log)
        self.worker.finished_log.connect(self.append_log)
        self.worker.progress_log.connect(self.append_log)
        self.worker.error.connect(self.on_error)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_browse_input.setEnabled(not running)
        self.btn_browse_output.setEnabled(not running)
        self.combo_algo.setEnabled(not running)
        self.spin_minsup.setEnabled(not running)
        self.spin_maxlen.setEnabled(not running)
        self.spin_minlen.setEnabled(not running)
        self.progress.setRange(0, 0 if running else 1)  # busy vs idle
        self.lbl_status.setText("Running…" if running else "Idle")

    def append_log(self, msg: str):
        self.txt_log.append(msg.rstrip())

    def on_error(self, msg: str):
        self.set_running(False)
        self.append_log(msg)
        QMessageBox.critical(self, "Error", msg)

    def on_done(self, total_ms: int, count: int):
        self.set_running(False)
        self.lbl_elapsed.setText(f"Elapsed: {total_ms} ms")
        self.lbl_count.setText(f"Patterns: {count}")
        self.append_log(f"Elapsed: {total_ms} ms; Patterns: {count}\n")
        self.append_log("Done. Output written.")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
