"""Exercise the real notebook loop without loading weights or doing local training."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("batches,expected", [(3, [2, 2]), (4, [2, 2, 2, 2])])
def test_an_incomplete_accumulation_group_does_not_reach_the_next_epoch(tmp_path, batches, expected):
    tree = ast.parse(Path("notebooks/pair_judge_notebook.py").read_text(encoding="utf-8"))
    loop = next(node for node in tree.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "epoch")
    gradients, steps = [], []
    optimizer, scaler, model = MagicMock(), MagicMock(), MagicMock()
    optimizer.zero_grad.side_effect = lambda **kwargs: gradients.clear()
    scaler.scale.return_value.backward.side_effect = lambda: gradients.append(1)
    scaler.step.side_effect = lambda optimizer: steps.append(len(gradients))
    model.return_value.loss = 1.0
    metrics = {"accuracy": 1.0, "macro_f1": 1.0, "counting_error": {"rate": 0.0},
               "negated_hypothesis": {"accuracy": 1.0}}
    namespace = dict(CFG={"epochs": 2, "grad_accum": 2, "max_grad_norm": 1.0, "time_budget_h": 5},
                     optimizer=optimizer, scaler=scaler, model=model, torch=MagicMock(), DEVICE="cpu",
                     loader=[{key: MagicMock() for key in ("input_ids", "attention_mask", "labels")} for _ in range(batches)],
                     schedule=MagicMock(), step=0, total_steps=len(expected), started=0,
                     time=SimpleNamespace(time=lambda: 0), log=[], best={"macro_f1": -1.0}, WORK=tmp_path,
                     evaluate=lambda *args, **kwargs: (metrics, None))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "<training-loop>", "exec"), namespace)
    assert steps == expected
    assert namespace["step"] == len(expected)
