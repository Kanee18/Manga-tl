"""The settings panel.

Only the settings whose right value depends on the material being translated
live here -- the model, which way the pages are read, how the output is
written, and the glossary. The measured thresholds stay in config.yaml, next
to the measurements that set them, because changing one of those without a
fresh measurement is how they regress.

Reading order earns its place at the top: it is source-dependent, changes
nothing about how a page looks if it is wrong, and quietly degrades every
translation on the page by feeding the model its context out of order.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from ..config import Config
from ..languages import PAIRS
from .theme import Palette


class SettingsDialog(QDialog):
    """Collects the run-affecting settings, and returns them as a mapping."""

    def __init__(self, cfg: Config, models: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self._cfg = cfg

        # Top of the panel, because it is the setting every other one is
        # downstream of: it decides the prompt, what counts as an untranslated
        # line, and how recognised lines are joined back together.
        self._pair = QComboBox()
        for pair in PAIRS:
            self._pair.addItem(pair.label, pair.code)
        current = self._pair.findData(cfg.translate.pair)
        self._pair.setCurrentIndex(current if current >= 0 else 0)
        self._pair.currentIndexChanged.connect(self._name_glossary_columns)
        self._pair.setToolTip(
            "Which way the translation goes. The recogniser reads all three "
            "source scripts,\n"
            "including vertical Japanese and Chinese."
        )

        self._model = QComboBox()
        # Editable, so a model that has not been pulled yet can still be typed
        # in. That is also what made the list look empty: an editable combo
        # with no visible arrow is indistinguishable from a text box, so
        # clicking it put a caret in the field and never opened anything. The
        # arrow is drawn now, and the count below says what is in the list
        # whether or not anyone opens it.
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        completer = self._model.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self._models_found = QLabel("")
        self._models_found.setStyleSheet("font-size: 11px;")
        self._refresh = QPushButton("Refresh")
        self._refresh.setToolTip("Ask Ollama again for the list of models")
        self._refresh.clicked.connect(self._reload_models)
        self._set_models(models)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 1.5)
        self._temperature.setSingleStep(0.05)
        self._temperature.setDecimals(2)
        self._temperature.setValue(cfg.translate.temperature)
        self._temperature.setToolTip(
            "Low but not zero. At zero the model repeats one bad phrasing "
            "across every balloon on a page."
        )

        self._page_batch = QCheckBox("Translate a whole page in one request")
        self._page_batch.setChecked(cfg.translate.page_batch)
        self._page_batch.setToolTip(
            "Keeps pronouns and register consistent between balloons, because "
            "the model sees the page as one conversation."
        )

        self._order = QComboBox()
        self._order.addItem("Right to left (unflipped manga)", "rtl")
        self._order.addItem("Left to right (mirrored release)", "ltr")
        self._order.setCurrentIndex(0 if cfg.detect.reading_order == "rtl" else 1)

        self._device = QComboBox()
        for label, value in [("Automatic", "auto"), ("GPU (CUDA)", "cuda"), ("CPU", "cpu")]:
            self._device.addItem(label, value)
        self._device.setCurrentIndex(
            ["auto", "cuda", "cpu"].index(cfg.models.device)
        )

        self._uppercase = QCheckBox("Set lettering in capitals")
        self._uppercase.setChecked(cfg.typeset.uppercase)
        self._growth = QDoubleSpinBox()
        self._growth.setRange(1.0, 2.0)
        self._growth.setSingleStep(0.05)
        self._growth.setDecimals(2)
        self._growth.setValue(cfg.typeset.max_growth_over_source)
        self._growth.setToolTip(
            "How much larger replacement lettering may be than the lettering "
            "it replaces. Keeps a short translation from coming back in "
            "display type just because the balloon had room."
        )

        self._suffix = QLineEdit(cfg.output.suffix)
        self._format = QComboBox()
        self._format.addItems(["png", "jpg"])
        self._format.setCurrentText(cfg.output.format)
        self._debug = QCheckBox("Write per-stage overlays to debug/")
        self._debug.setChecked(cfg.output.debug)

        self._glossary = QTableWidget(0, 2)
        self._name_glossary_columns()
        self._glossary.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._glossary.verticalHeader().setVisible(False)
        for source, target in sorted(cfg.translate.glossary.items()):
            self._add_term(source, target)

        self._build(cfg)

    def _name_glossary_columns(self) -> None:
        """Head the glossary with the pair actually being translated.

        Left at "English / Indonesian" it tells a Japanese-to-English user to
        enter their terms the wrong way round, which is worse than saying
        nothing: the glossary is injected into the prompt as `source -> target`
        and a reversed entry steers the model backwards.
        """
        pair = PAIRS[max(0, self._pair.currentIndex())]
        self._glossary.setHorizontalHeaderLabels(
            [pair.source.name, pair.target.name]
        )

    # -- models ----------------------------------------------------------

    def _set_models(self, models: list[str]) -> None:
        """Fill the list, keeping whatever is currently chosen selected."""
        chosen = self._model.currentText().strip() or self._cfg.translate.model
        blocked = self._model.blockSignals(True)
        try:
            self._model.clear()
            self._model.addItems(models)
            # A model set in config but not installed still has to be
            # selectable, or opening the dialog would silently change it.
            if chosen and chosen not in models:
                self._model.addItem(chosen)
            self._model.setCurrentText(chosen)
        finally:
            self._model.blockSignals(blocked)

        if models:
            self._models_found.setText(
                f"{len(models)} model{'s' if len(models) != 1 else ''} installed"
            )
            self._models_found.setStyleSheet(
                f"color: {Palette.faint}; font-size: 11px;"
            )
            self._model.setToolTip("Models Ollama reports as installed")
        else:
            self._models_found.setText(
                "Ollama did not answer — start it with `ollama serve`, then Refresh"
            )
            self._models_found.setStyleSheet(
                f"color: {Palette.warn}; font-size: 11px;"
            )
            self._model.setToolTip(
                "Ollama did not answer, so this list is empty. "
                "Start it with `ollama serve` and press Refresh."
            )

    def _reload_models(self) -> None:
        """Re-query Ollama, so a server started late does not need a restart."""
        from ..pipeline.translate import OllamaTranslator

        self._refresh.setEnabled(False)
        self._models_found.setText("Asking Ollama...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            with OllamaTranslator(self._cfg) as translator:
                self._set_models(translator.list_models())
        except Exception:  # noqa: BLE001 -- reported in the label above
            self._set_models([])
        finally:
            QApplication.restoreOverrideCursor()
            self._refresh.setEnabled(True)

    # -- layout ----------------------------------------------------------

    def _build(self, cfg: Config) -> None:
        translation = QFormLayout()
        translation.addRow("Translate", self._pair)

        picker = QHBoxLayout()
        picker.setSpacing(6)
        picker.addWidget(self._model, 1)
        picker.addWidget(self._refresh)
        translation.addRow("Model", picker)
        translation.addRow("", self._models_found)
        translation.addRow("Temperature", self._temperature)
        translation.addRow("", self._page_batch)
        translation_box = QGroupBox("Translation")
        translation_box.setLayout(translation)

        source = QFormLayout()
        source.addRow("Reading order", self._order)
        source.addRow("Run models on", self._device)
        source_box = QGroupBox("Source and hardware")
        source_box.setLayout(source)

        lettering = QFormLayout()
        lettering.addRow("", self._uppercase)
        lettering.addRow("Max growth over source", self._growth)
        lettering_box = QGroupBox("Lettering")
        lettering_box.setLayout(lettering)

        output = QFormLayout()
        output.addRow("Output folder suffix", self._suffix)
        output.addRow("Image format", self._format)
        output.addRow("", self._debug)
        output_box = QGroupBox("Output")
        output_box.setLayout(output)

        add = QPushButton("Add term")
        add.clicked.connect(lambda: self._add_term("", ""))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_term)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)

        glossary = QVBoxLayout()
        glossary.addWidget(
            QLabel(
                "Terms the translator must always render this way -- names, "
                "titles, and anything a series is particular about."
            )
        )
        glossary.addWidget(self._glossary)
        glossary.addLayout(buttons)
        glossary_box = QGroupBox("Glossary")
        glossary_box.setLayout(glossary)

        actions = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)

        body = QVBoxLayout()
        body.addWidget(translation_box)
        body.addWidget(source_box)
        body.addWidget(lettering_box)
        body.addWidget(output_box)
        body.addWidget(glossary_box, 1)
        body.addWidget(
            QLabel(
                "Saved to settings.local.yaml. Detection and fitting "
                "thresholds stay in config.yaml, beside their measurements."
            )
        )
        body.addWidget(actions)
        self.setLayout(body)

    def _add_term(self, source: str, target: str) -> None:
        row = self._glossary.rowCount()
        self._glossary.insertRow(row)
        self._glossary.setItem(row, 0, QTableWidgetItem(source))
        self._glossary.setItem(row, 1, QTableWidgetItem(target))

    def _remove_term(self) -> None:
        for index in sorted(
            {i.row() for i in self._glossary.selectedIndexes()}, reverse=True
        ):
            self._glossary.removeRow(index)

    # -- result ----------------------------------------------------------

    def overrides(self) -> dict[str, Any]:
        """The chosen settings, shaped like config.yaml."""
        glossary: dict[str, str] = {}
        for row in range(self._glossary.rowCount()):
            source = self._glossary.item(row, 0)
            target = self._glossary.item(row, 1)
            if source and target and source.text().strip() and target.text().strip():
                glossary[source.text().strip()] = target.text().strip()

        return {
            "models": {"device": self._device.currentData()},
            "detect": {"reading_order": self._order.currentData()},
            "translate": {
                "pair": self._pair.currentData(),
                "model": self._model.currentText().strip(),
                "temperature": round(self._temperature.value(), 3),
                "page_batch": self._page_batch.isChecked(),
                "glossary": glossary,
            },
            "typeset": {
                "uppercase": self._uppercase.isChecked(),
                "max_growth_over_source": round(self._growth.value(), 3),
            },
            "output": {
                "suffix": self._suffix.text().strip() or "_ID",
                "format": self._format.currentText(),
                "debug": self._debug.isChecked(),
            },
        }
