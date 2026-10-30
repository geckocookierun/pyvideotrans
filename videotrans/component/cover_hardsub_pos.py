"""Popup to place the source-hardsub cover box on the imported video."""
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import QRect, Qt, QPoint
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from videotrans.configure.config import tr
from videotrans.configure.constants import VIDEO_EXTS
from videotrans.task._stage_assemble import cover_hardsub_geometry


def cover_button_text(x, y, w, h) -> str:
    yp = int(round(y * 100))
    hp = int(round(h * 100))
    if x > 0.005 or w < 0.995:
        return f"X {int(round(x * 100))}%  Y {yp}%  ·  {int(round(w * 100))}×{hp}%"
    return f"Y {yp}% · H {hp}%"


def apply_cover_box(window, x, y, w, h) -> None:
    window.cover_hardsub_x = float(x)
    window.cover_hardsub_y = float(y)
    window.cover_hardsub_w = float(w)
    if hasattr(window, 'cover_hardsub_pct'):
        window.cover_hardsub_pct.setText(str(int(round(h * 100))))
    if hasattr(window, 'cover_hardsub_pos_btn'):
        window.cover_hardsub_pos_btn.setText(cover_button_text(x, y, w, h))


def cover_box_from_window(window) -> dict:
    raw = window.cover_hardsub_pct.text() if hasattr(window, 'cover_hardsub_pct') else 0.14
    x, y, w, h = cover_hardsub_geometry(
        raw,
        getattr(window, 'cover_hardsub_x', 0),
        getattr(window, 'cover_hardsub_y', -1),
        getattr(window, 'cover_hardsub_w', 1),
    )
    return {
        'cover_source_hardsub_pct': h,
        'cover_hardsub_x': x,
        'cover_hardsub_y': y,
        'cover_hardsub_w': w,
    }


def _imported_video(window) -> str:
    action = getattr(window, 'win_action', None)
    queue = list(getattr(action, 'queue_mp4', None) or [])
    for path in queue:
        p = Path(str(path))
        if p.is_file() and p.suffix.lower().lstrip('.') in VIDEO_EXTS:
            return p.as_posix()
    return ''


class _CoverCanvas(QWidget):
    """Video frame with a draggable cover rectangle. box is x,y,w,h in 0–1."""

    def __init__(self, box, on_change, parent=None):
        super().__init__(parent)
        self.box = list(box)
        self._on_change = on_change
        self._pix = QPixmap()
        self._drag = None
        self._grab = QPoint()
        self._start_box = None
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def set_frame(self, pix: QPixmap) -> None:
        self._pix = pix or QPixmap()
        self.update()

    def set_box(self, box) -> None:
        self.box = list(box)
        self.update()

    def _frame_rect(self) -> QRect:
        avail = self.rect().adjusted(8, 8, -8, -8)
        if self._pix.isNull():
            side_w = avail.width()
            side_h = int(side_w * 9 / 16)
            if side_h > avail.height():
                side_h = avail.height()
                side_w = int(side_h * 16 / 9)
            x = avail.x() + (avail.width() - side_w) // 2
            y = avail.y() + (avail.height() - side_h) // 2
            return QRect(x, y, max(side_w, 1), max(side_h, 1))
        scaled = self._pix.size().scaled(avail.size(), Qt.KeepAspectRatio)
        x = avail.x() + (avail.width() - scaled.width()) // 2
        y = avail.y() + (avail.height() - scaled.height()) // 2
        return QRect(x, y, max(scaled.width(), 1), max(scaled.height(), 1))

    def _band_rect(self) -> QRect:
        frame = self._frame_rect()
        x, y, w, h = self.box
        return QRect(
            int(frame.x() + x * frame.width()),
            int(frame.y() + y * frame.height()),
            max(6, int(w * frame.width())),
            max(6, int(h * frame.height())),
        )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        frame = self._frame_rect()
        painter.fillRect(self.rect(), QColor('#1e1e1e'))
        if self._pix.isNull():
            painter.fillRect(frame, QColor('#2a2a2a'))
        else:
            painter.drawPixmap(frame, self._pix)
        band = self._band_rect()
        painter.fillRect(band, QColor(0, 0, 0, 150))
        painter.setPen(QPen(QColor('#ffcc00'), 2))
        painter.drawRect(band.adjusted(0, 0, -1, -1))
        painter.end()

    def _hit(self, pos: QPoint) -> str:
        band = self._band_rect()
        if not band.contains(pos):
            return ''
        margin = 8
        on_top = abs(pos.y() - band.top()) <= margin
        on_bottom = abs(pos.y() - band.bottom()) <= margin
        on_left = abs(pos.x() - band.left()) <= margin
        on_right = abs(pos.x() - band.right()) <= margin
        if on_bottom:
            return 's'
        if on_top:
            return 'n'
        if on_right:
            return 'e'
        if on_left:
            return 'w'
        return 'move'

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        kind = self._hit(event.position().toPoint())
        if not kind:
            return
        self._drag = kind
        self._grab = event.position().toPoint()
        self._start_box = list(self.box)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if not self._drag:
            kind = self._hit(pos)
            self.setCursor({
                's': Qt.SizeVerCursor,
                'n': Qt.SizeVerCursor,
                'e': Qt.SizeHorCursor,
                'w': Qt.SizeHorCursor,
                'move': Qt.SizeAllCursor,
            }.get(kind, Qt.ArrowCursor))
            return
        frame = self._frame_rect()
        dx = (pos.x() - self._grab.x()) / frame.width()
        dy = (pos.y() - self._grab.y()) / frame.height()
        x, y, w, h = self._start_box
        if self._drag == 'move':
            x += dx
            y += dy
        elif self._drag == 's':
            h += dy
        elif self._drag == 'n':
            y += dy
            h -= dy
        elif self._drag == 'e':
            w += dx
        elif self._drag == 'w':
            x += dx
            w -= dx
        h = min(0.60, max(0.03, h))
        w = min(1.0, max(0.05, w))
        x = min(max(0.0, x), 1.0 - w)
        y = min(max(0.0, y), 1.0 - h)
        self.box = [x, y, w, h]
        self.update()
        self._on_change(self.box, from_canvas=True)

    def mouseReleaseEvent(self, _event):
        self._drag = None
        self._start_box = None


class CoverHardsubPosDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Cover position'))
        self.resize(720, 640)
        self._window = parent
        self._video = _imported_video(parent)
        self._duration_ms = 0
        self._frame_path = ''
        box = cover_box_from_window(parent)
        self._syncing = False
        self._box = [
            box['cover_hardsub_x'],
            box['cover_hardsub_y'],
            box['cover_hardsub_w'],
            box['cover_source_hardsub_pct'],
        ]

        root = QVBoxLayout(self)
        hint = QLabel(tr(
            'Drag the box on the imported video. Y is the distance from the top; height is how tall the blur is.'
        ))
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.file_label = QLabel(self._video or tr('No video imported. Import a video to preview the frame.'))
        self.file_label.setWordWrap(True)
        root.addWidget(self.file_label)

        self.canvas = _CoverCanvas(self._box, self._from_canvas)
        root.addWidget(self.canvas, 1)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel(tr('Preview time')))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setEnabled(False)
        self.time_slider.sliderReleased.connect(self._seek)
        time_row.addWidget(self.time_slider, 1)
        self.time_label = QLabel('0:00')
        time_row.addWidget(self.time_label)
        root.addLayout(time_row)

        spins = QHBoxLayout()
        self.spin_x = self._spin(tr('From left'), 0, 95)
        self.spin_y = self._spin(tr('From top'), 0, 97)
        self.spin_w = self._spin(tr('Cover width'), 5, 100)
        self.spin_h = self._spin(tr('Cover height %'), 3, 60)
        for label, spin in (
            (tr('From left'), self.spin_x),
            (tr('From top'), self.spin_y),
            (tr('Cover width'), self.spin_w),
            (tr('Cover height %'), self.spin_h),
        ):
            spins.addWidget(QLabel(label))
            spins.addWidget(spin)
        root.addLayout(spins)

        bottom_btn = QPushButton(tr('Stick cover to bottom'))
        bottom_btn.clicked.connect(self._stick_bottom)
        root.addWidget(bottom_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._push_spins()
        self._load_video()

    def _spin(self, _label, lo, hi) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSuffix('%')
        spin.valueChanged.connect(self._from_spins)
        return spin

    def _push_spins(self) -> None:
        x, y, w, h = self._box
        self._syncing = True
        self.spin_x.setValue(int(round(x * 100)))
        self.spin_y.setValue(int(round(y * 100)))
        self.spin_w.setValue(int(round(w * 100)))
        self.spin_h.setValue(int(round(h * 100)))
        self._syncing = False

    def _from_canvas(self, box, from_canvas=False) -> None:
        self._box = list(box)
        if from_canvas:
            self._push_spins()

    def _from_spins(self) -> None:
        if self._syncing:
            return
        stuck_bottom = abs((self._box[1] + self._box[3]) - 1.0) < 0.02
        h = min(0.60, max(0.03, self.spin_h.value() / 100))
        w = min(1.0, max(0.05, self.spin_w.value() / 100))
        x = min(max(0.0, self.spin_x.value() / 100), 1.0 - w)
        y = min(max(0.0, self.spin_y.value() / 100), 1.0 - h)
        if stuck_bottom and self.sender() is self.spin_h:
            y = 1.0 - h
        self._box = [x, y, w, h]
        self.canvas.set_box(self._box)
        self._push_spins()

    def _stick_bottom(self) -> None:
        h = self._box[3]
        self._box = [0.0, 1.0 - h, 1.0, h]
        self.canvas.set_box(self._box)
        self._push_spins()

    def box(self):
        return tuple(self._box)

    def _load_video(self) -> None:
        if not self._video:
            return
        try:
            from videotrans.util.help_ffmpeg import get_video_duration
            self._duration_ms = int(get_video_duration(self._video) or 0)
        except Exception:
            self._duration_ms = 0
        self.time_slider.setEnabled(self._duration_ms > 500)
        start = 2000 if self._duration_ms > 4000 else 0
        if self._duration_ms:
            self.time_slider.setValue(int(start * 1000 / self._duration_ms))
        self._grab(start / 1000)

    def _seek(self) -> None:
        if not self._duration_ms:
            return
        sec = self.time_slider.value() / 1000 * (self._duration_ms / 1000)
        self._grab(sec)

    def _grab(self, sec: float) -> None:
        if not self._video:
            return
        mins, secs = divmod(int(sec), 60)
        self.time_label.setText(f'{mins}:{secs:02d}')
        dest = Path(tempfile.gettempdir()) / f'cover-preview-{id(self)}.jpg'
        self._frame_path = dest.as_posix()
        try:
            subprocess.run(
                [
                    'ffmpeg', '-hide_banner', '-nostdin', '-y',
                    '-ss', f'{max(0.0, sec):.3f}',
                    '-i', self._video,
                    '-frames:v', '1',
                    '-q:v', '3',
                    self._frame_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            pix = QPixmap(self._frame_path)
            if not pix.isNull():
                self.canvas.set_frame(pix)
        except Exception:
            self.file_label.setText(tr('Could not read a frame from the imported video.'))

    def done(self, code):
        if self._frame_path:
            Path(self._frame_path).unlink(missing_ok=True)
        super().done(code)


def open_cover_hardsub_dialog(window) -> None:
    if getattr(window, '_cover_dialog_open', False):
        return
    window._cover_dialog_open = True
    try:
        dialog = CoverHardsubPosDialog(window)
        if dialog.exec() != QDialog.Accepted:
            return
        x, y, w, h = dialog.box()
        apply_cover_box(window, x, y, w, h)
        if hasattr(window, 'cover_source_hardsub'):
            window.cover_source_hardsub.setChecked(True)
    finally:
        window._cover_dialog_open = False
