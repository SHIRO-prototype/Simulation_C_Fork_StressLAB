"""Rich Live Panel - real-time instrument display for StressLAB simulations.

Displays a continuously updating dashboard showing:
  - Run metadata (run_id, seed, dt, horizon, versions)
  - Current estimates (time_to_TCA, miss distance, Pc, staleness, etc.)
  - Decision states side-by-side (threshold-v1 vs integrity-v1)

Uses Rich Live + Layout + Panel with ~2Hz refresh.
Falls back to periodic single-line logs in non-interactive terminals.
"""

from __future__ import annotations

from typing import Optional

from stresslab import __version__ as STRESSLAB_VERSION
from stresslab.metrics_contract import METRICS_CONTRACT_VERSION
from stresslab.modules.logging_engine import SCHEMA_VERSION


def _is_interactive() -> bool:
    """Check if stdout is connected to an interactive terminal."""
    import sys
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class LivePanelCallback:
    """Callback for simulation runner that drives a Rich live display.

    Attach this as the ``step_callback`` in ``run_simulation()`` to get
    a continuously updating instrument panel in the terminal.
    """

    def __init__(
        self,
        run_id: str,
        seed: int,
        dt: float,
        t_end: float,
        dynamics: str,
        refresh_per_second: float = 2.0,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.dt = dt
        self.t_end = t_end
        self.dynamics = dynamics
        self.refresh_per_second = refresh_per_second

        self._interactive = _is_interactive()
        self._live = None
        self._last_data: dict = {}
        self._step_count = 0
        self._fallback_interval = max(1, int(1.0 / (dt * refresh_per_second))) if dt > 0 else 50

    def start(self) -> None:
        """Initialize the live display (call before simulation loop)."""
        if self._interactive:
            try:
                from rich.live import Live
                from rich.console import Console
                console = Console()
                self._live = Live(
                    self._build_layout(),
                    console=console,
                    refresh_per_second=self.refresh_per_second,
                    transient=True,
                )
                self._live.start()
            except ImportError:
                self._interactive = False

    def stop(self) -> None:
        """Stop the live display (call after simulation loop)."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def on_step(self, step_data: dict) -> None:
        """Called by the simulation runner after each timestep.

        Args:
            step_data: dict with keys from the current timestep:
                t, time_to_tca, miss_distance, rel_velocity,
                staleness, pc_reference, pc_degraded, risk_ratio,
                cov_norm, growth_rate, threshold_v1_state,
                integrity_v1_state, integrity_v1_score
        """
        self._last_data = step_data
        self._step_count += 1

        if self._interactive and self._live is not None:
            self._live.update(self._build_layout())
        elif self._step_count % self._fallback_interval == 0:
            self._print_fallback_line()

    def _print_fallback_line(self) -> None:
        """Print a single-line status for non-interactive terminals."""
        d = self._last_data
        if not d:
            return
        t = d.get("t", 0.0)
        pct = (t / self.t_end * 100) if self.t_end > 0 else 0.0
        print(
            f"[StressLAB] t={t:.0f}s ({pct:.0f}%) | "
            f"miss={d.get('miss_distance', 0):.4f}km | "
            f"Pc_d={d.get('pc_degraded', 0):.2e} | "
            f"ratio={d.get('risk_ratio', 0):.3f} | "
            f"stale={d.get('staleness', 0):.0f}s | "
            f"T-v1={d.get('threshold_v1_state', '?')} | "
            f"I-v1={d.get('integrity_v1_state', '?')}"
        )

    def _build_layout(self):
        """Build the Rich Layout with all panels."""
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.align import Align

        d = self._last_data
        t = d.get("t", 0.0)
        pct = (t / self.t_end * 100) if self.t_end > 0 else 0.0

        # ---- Header panel ----
        header_table = Table.grid(padding=(0, 2))
        header_table.add_column(justify="left")
        header_table.add_column(justify="left")
        header_table.add_column(justify="left")
        header_table.add_column(justify="left")
        header_table.add_row(
            f"[bold cyan]Run:[/] {self.run_id}",
            f"[bold cyan]Seed:[/] {self.seed}",
            f"[bold cyan]dt:[/] {self.dt}s",
            f"[bold cyan]Horizon:[/] {self.t_end:.0f}s",
        )
        header_table.add_row(
            f"[bold cyan]Dynamics:[/] {self.dynamics}",
            f"[bold cyan]Schema:[/] {SCHEMA_VERSION}",
            f"[bold cyan]Contract:[/] {METRICS_CONTRACT_VERSION}",
            f"[bold cyan]Code:[/] {STRESSLAB_VERSION}",
        )

        # ---- Progress bar ----
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "[green]" + "=" * filled + "[/]" + " " * (bar_len - filled)
        progress_text = f"  [{bar}] {pct:.1f}%  t={t:.0f}s / {self.t_end:.0f}s  ({self._step_count} steps)"

        # ---- Estimates panel ----
        est = Table(title="Current Estimates", box=None, show_header=True,
                    header_style="bold magenta")
        est.add_column("Parameter", style="cyan", min_width=22)
        est.add_column("Value", justify="right", min_width=16)

        est.add_row("Time to TCA", f"{d.get('time_to_tca', 0):.1f} s")
        est.add_row("Miss distance", f"{d.get('miss_distance', 0):.6f} km")
        est.add_row("V_rel", f"{d.get('rel_velocity', 0):.4f} km/s")
        est.add_row("Staleness", f"{d.get('staleness', 0):.0f} s")
        est.add_row("Pc (reference)", f"{d.get('pc_reference', 0):.3e}")
        est.add_row("Pc (degraded)", f"{d.get('pc_degraded', 0):.3e}")
        est.add_row("Risk ratio", f"{d.get('risk_ratio', 0):.4f}")
        est.add_row("Cov norm", f"{d.get('cov_norm', 0):.4e}")
        est.add_row("Growth rate", f"{d.get('growth_rate', 0):.4e}")

        # ---- Decision panel ----
        dec = Table(title="Decision Models", box=None, show_header=True,
                    header_style="bold magenta")
        dec.add_column("", style="cyan", min_width=18)
        dec.add_column("Threshold-v1", justify="center", min_width=14)
        dec.add_column("Integrity-v1", justify="center", min_width=14)

        tv1 = d.get("threshold_v1_state", "---")
        iv1 = d.get("integrity_v1_state", "---")

        # Color-code states
        tv1_styled = _style_alert(tv1)
        iv1_styled = _style_integrity(iv1)

        dec.add_row("State", tv1_styled, iv1_styled)
        dec.add_row(
            "Score",
            "---",
            f"{d.get('integrity_v1_score', 0):.4f}",
        )
        dec.add_row(
            "Trigger time",
            _fmt_trigger(d.get("threshold_v1_trigger")),
            _fmt_trigger(d.get("integrity_v1_trigger")),
        )

        # ---- Compose layout ----
        layout = Layout()
        layout.split_column(
            Layout(Panel(header_table, title="[bold]StressLAB Instrument Panel[/]",
                         border_style="blue"), name="header", size=5),
            Layout(Text(progress_text), name="progress", size=1),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(Panel(est, border_style="green"), name="estimates"),
            Layout(Panel(dec, border_style="yellow"), name="decisions"),
        )

        return layout


def _style_alert(state: str) -> str:
    """Apply Rich styling to threshold-v1 alert state."""
    if state == "Alert":
        return "[bold red]Alert[/]"
    if state == "Safe":
        return "[bold green]Safe[/]"
    return state


def _style_integrity(state: str) -> str:
    """Apply Rich styling to integrity-v1 state."""
    styles = {
        "Monitor": "[bold green]Monitor[/]",
        "Watch": "[bold yellow]Watch[/]",
        "Warning": "[bold dark_orange]Warning[/]",
        "Critical": "[bold red]Critical[/]",
    }
    return styles.get(state, state)


def _fmt_trigger(t) -> str:
    """Format trigger time for display."""
    if t is None:
        return "---"
    return f"{t:.1f} s"
