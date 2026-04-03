"""Real-time loss plot widget using pyqtgraph."""

import pyqtgraph as pg


class LossPlot(pg.PlotWidget):
    """Real-time training loss chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("w")
        self.setTitle("Training Loss", color="k")
        self.setLabel("left", "Loss", color="k")
        self.setLabel("bottom", "Iteration", color="k")
        self.showGrid(x=True, y=True, alpha=0.3)

        self._iterations = []
        self._losses = []
        self._curve = self.plot([], [], pen=pg.mkPen("b", width=2))

    def add_point(self, iteration, loss):
        """Add a data point and update the plot."""
        self._iterations.append(iteration)
        self._losses.append(loss)
        self._curve.setData(self._iterations, self._losses)

    def clear_plot(self):
        """Reset the plot data."""
        self._iterations.clear()
        self._losses.clear()
        self._curve.setData([], [])
