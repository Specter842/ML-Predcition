"""Phase 4 stretch goal: a small LSTM — behind a gate that must open first.

The brief is explicit that deep learning is only permitted *after* a tree or
linear model has already beaten the phase-3 baseline on the same folds. That
constraint is enforced here in code rather than left to discipline, because
"let's just try an LSTM and see" is precisely the failure mode it exists to
prevent.

:func:`gate_status` inspects the backtest results and reports whether the gate
is open. :class:`LSTMModel` refuses to fit unless it is handed a passing
:class:`GateResult`. If the gate never opens — which, on ~300 monthly
observations, is the likely outcome — that is the finding, and the honest thing
is to report it rather than route around it.

Sizing, if it ever does run: one layer, small hidden state, dropout, early
stopping on a chronological validation tail. A sequence model with more
parameters than the sample has observations is not a forecast, it is a lookup
table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.base import BaseModel, NumericPreprocessor

#: Models whose success can open the gate (phase-3 baselines are excluded —
#: beating a baseline with another baseline proves nothing).
GATE_ELIGIBLE = {"ridge", "elastic_net", "ridge_fixed", "enet_fixed", "xgboost", "lightgbm"}

#: The gate must be cleared against all of these, not just the random walk.
GATE_BENCHMARKS = ("random_walk", "atkeson_ohanian")


@dataclass(frozen=True)
class GateResult:
    """Whether phase 4 has earned the right to try deep learning."""

    passed: bool
    reason: str
    winners: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.passed


def gate_status(results: pd.DataFrame, alpha: float = 0.05) -> GateResult:
    """Has any tree/linear model beaten the baselines at ``alpha``, on the same folds?

    Requires, for at least one eligible model and horizon: RMSE below the
    benchmark's *and* a significant Diebold-Mariano p-value, against every
    benchmark in :data:`GATE_BENCHMARKS`.
    """
    if results.empty:
        return GateResult(False, "no backtest results to evaluate")

    eligible = results[results["model"].isin(GATE_ELIGIBLE)]
    if eligible.empty:
        return GateResult(
            False, "no tree or linear challenger has been backtested yet"
        )

    winners: list[str] = []
    for rec in eligible.itertuples(index=False):
        row = rec._asdict()
        cleared = True
        for bench in GATE_BENCHMARKS:
            ratio = row.get(f"rmse_ratio_vs_{bench}")
            pval = row.get(f"dm_p_vs_{bench}")
            if ratio is None or pval is None or not np.isfinite(ratio) or not np.isfinite(pval):
                cleared = False
                break
            if not (ratio < 1.0 and pval < alpha):
                cleared = False
                break
        if cleared:
            winners.append(f"{row['model']}@{int(row['as_of_lag_days'])}d")

    if winners:
        return GateResult(
            True,
            f"cleared by {', '.join(winners)} against {', '.join(GATE_BENCHMARKS)} at p<{alpha}",
            tuple(winners),
        )
    return GateResult(
        False,
        f"no tree/linear model beat {' and '.join(GATE_BENCHMARKS)} at p<{alpha} — "
        "per the brief, deep learning stays unbuilt until one does",
    )


class GateClosed(RuntimeError):
    """Raised when a deep model is asked to fit before the gate opens."""


class LSTMModel(BaseModel):
    """Single-layer LSTM over a short window of the feature table.

    Not instantiated by the default pipeline. Requires ``torch`` and a passing
    :class:`GateResult`.
    """

    name = "lstm"
    note = "1-layer LSTM, gated on a tree/linear model beating the baselines first"

    def __init__(
        self,
        gate: GateResult,
        *,
        sequence_length: int = 12,
        hidden_size: int = 16,
        dropout: float = 0.2,
        epochs: int = 200,
        patience: int = 20,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        if not gate.passed:
            raise GateClosed(
                "LSTM is gated behind a tree/linear model first beating the phase-3 "
                f"baselines on the same folds. Gate is closed: {gate.reason}"
            )
        self.gate = gate
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.seed = seed
        self.pre = NumericPreprocessor(standardise=True)
        self.model = None
        self._fallback = 0.0

    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa: F401
            from torch import nn  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LSTMModel needs PyTorch — `pip install torch`. It is intentionally "
                "not in requirements.txt, since the gate rarely opens on this sample."
            ) from exc
        import torch

        return torch

    def _sequences(self, design: np.ndarray, target: np.ndarray | None):
        """Stack rows into overlapping windows, oldest first within each window."""
        n, k = design.shape
        length = min(self.sequence_length, n)
        xs, ys = [], []
        for end in range(length, n + 1):
            xs.append(design[end - length : end])
            if target is not None:
                ys.append(target[end - 1])
        if not xs:
            return np.zeros((0, length, k), dtype="float32"), np.zeros(0, dtype="float32")
        return (
            np.asarray(xs, dtype="float32"),
            np.asarray(ys, dtype="float32") if target is not None else np.zeros(0, dtype="float32"),
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> "LSTMModel":
        torch = self._require_torch()
        from torch import nn

        target = y.to_numpy(dtype="float64")
        ok = np.isfinite(target)
        self._fallback = float(np.mean(target[ok])) if ok.any() else 0.0
        if ok.sum() < 100:
            self.model = None
            return self

        torch.manual_seed(self.seed)
        design = self.pre.fit_transform(X.loc[ok])
        seq_x, seq_y = self._sequences(design, target[ok])
        if len(seq_x) < 60:
            self.model = None
            return self

        # Chronological validation tail for early stopping — never random.
        split = int(len(seq_x) * 0.8)
        tr_x = torch.from_numpy(seq_x[:split])
        tr_y = torch.from_numpy(seq_y[:split]).unsqueeze(-1)
        va_x = torch.from_numpy(seq_x[split:])
        va_y = torch.from_numpy(seq_y[split:]).unsqueeze(-1)

        class Net(nn.Module):
            def __init__(self, n_features: int, hidden: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
                self.drop = nn.Dropout(dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :]))

        net = Net(seq_x.shape[2], self.hidden_size, self.dropout)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_loss, best_state, waited = float("inf"), None, 0
        for _ in range(self.epochs):
            net.train()
            optimiser.zero_grad()
            loss_fn(net(tr_x), tr_y).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(va_x), va_y)) if len(va_x) else float("inf")
            if val_loss < best_loss - 1e-6:
                best_loss, waited = val_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model = net
        self._train_tail = design[-(self.sequence_length - 1):] if self.sequence_length > 1 else None
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self._fallback, dtype="float64")
        torch = self._require_torch()

        design = self.pre.transform(X)
        # Prepend the tail of training so the first test rows have a full window.
        if getattr(self, "_train_tail", None) is not None and len(self._train_tail):
            design = np.vstack([self._train_tail, design])
            offset = len(self._train_tail)
        else:
            offset = 0

        seq_x, _ = self._sequences(design, None)
        if len(seq_x) == 0:
            return np.full(len(X), self._fallback, dtype="float64")
        with torch.no_grad():
            preds = self.model(torch.from_numpy(seq_x)).numpy().ravel()

        # Align: the last len(X) predictions correspond to the test rows.
        preds = preds[-len(X):] if len(preds) >= len(X) else np.pad(
            preds, (len(X) - len(preds), 0), constant_values=self._fallback
        )
        return np.where(np.isfinite(preds), preds, self._fallback).astype("float64")
