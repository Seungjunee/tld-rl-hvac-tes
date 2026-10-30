"""Padding with frozen features (`padding_frozen`, cases `*_smud_sacp_padff`).

The arm is the Padding fine-tuning with one change: the first hidden layer of
the actor and of both critics is fixed. These tests pin (1) that every padff
case resolves to its padrl twin except for that flag and the output names,
and (2) that the runner really freezes those layers, trains the rest, and
leaves the padrl path untouched.
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch
from stable_baselines3 import SAC

from studies.case_study.core import load_registry, resolve_case
from studies.case_study.core.types import ModelBundle
from studies.case_study.methods.padding_transfer import RUNNER

CONFIG_ROOT = "studies/case_study/config"
PAPER = ["office", "lbuilding", "warehouse", "mediumoffice"]
PAIRS = [(s, t) for s in PAPER for t in PAPER if s != t]
FIRST = ("actor.latent_pi.0", "critic.qf0.0", "critic.qf1.0")


def _flat(d, prefix=""):
    if not isinstance(d, dict):
        return {prefix: d}
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}.{k}"))
    return out


def test_padff_cases_are_padrl_twins():
    registry = load_registry(CONFIG_ROOT)
    assert registry.groups["r1_padff"] == [
        f"transfer_{s}_to_{t}_smud_sacp_padff" for s, t in PAIRS
    ]
    allowed = {
        ".id",
        ".env.env_name",
        ".export.destination",
        ".method.id",
        ".method.default_model_name",
        ".method.training.freeze_first_layer",
        ".outputs.model_name",
        ".training.freeze_first_layer",
    }
    for s, t in PAIRS:
        padrl = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_padrl", registry)
        padff = resolve_case(f"transfer_{s}_to_{t}_smud_sacp_padff", registry)
        fa, fb = _flat(asdict(padrl)), _flat(asdict(padff))
        diff = {k for k in set(fa) | set(fb) if str(fa.get(k)) != str(fb.get(k))}
        assert diff <= allowed, (s, t, diff - allowed)
        assert padff.method.runner == "padding_transfer"
        assert padff.training["freeze_first_layer"] is True
        assert "freeze_first_layer" not in padrl.training
        assert padff.training["actor_lr"] == 1e-4
        assert padff.training["critic_lr"] == 3e-4


def _run(tmp_path: Path, freeze: bool):
    env = gym.make("Pendulum-v1")
    src = SAC("MlpPolicy", env, learning_starts=0, seed=0, verbose=0)
    src.learn(300)
    path = tmp_path / "source.zip"
    src.save(path)
    before = SAC.load(path).policy.state_dict()

    training = {"actor_lr": 1e-4, "critic_lr": 3e-4, "model_kwargs": {"seed": 42}}
    if freeze:
        training["freeze_first_layer"] = True
    case = SimpleNamespace(
        training=training,
        method=SimpleNamespace(id="padding_frozen" if freeze else "padding_transfer"),
    )
    services = SimpleNamespace(
        load_source_bundle=lambda c: ModelBundle(
            case_id="src", building_id="x", method_id="x", model_path=path
        ),
        load_model=lambda p, env=None: SAC.load(str(p), env=env),
        compute_total_timesteps=lambda c, e: 300,
        build_learn_kwargs=lambda c: {},
    )
    execution = RUNNER.execute(case, env, services)
    return before, execution


def _max_change(before, after, name):
    return max(
        float((after[f"{name}.{k}"] - before[f"{name}.{k}"]).abs().max())
        for k in ("weight", "bias")
    )


def test_frozen_runner_fixes_first_layers_and_trains_the_rest(tmp_path):
    before, execution = _run(tmp_path, freeze=True)
    model, md = execution.model, execution.metadata
    after = model.policy.state_dict()
    for name in FIRST:
        assert _max_change(before, after, name) == 0.0, name
    for name in ("actor.latent_pi.2", "actor.mu", "critic.qf0.2", "critic.qf1.4"):
        assert _max_change(before, after, name) > 0.0, name
    for i in range(2):
        tgt = model.critic_target.q_networks[i][0]
        onl = model.critic.q_networks[i][0]
        assert torch.allclose(tgt.weight, onl.weight, atol=1e-5)
        assert torch.allclose(tgt.bias, onl.bias, atol=1e-5)
    frozen_ids = {id(p) for name in FIRST for p in _module(model, name).parameters()}
    for opt in (model.actor.optimizer, model.critic.optimizer):
        for g in opt.param_groups:
            assert not frozen_ids & {id(p) for p in g["params"]}
    check = md["freeze_check"]
    assert check["frozen_max_abs_change"] == 0.0
    assert check["trainable_tensors_changed"] == check["trainable_tensors_total"]
    assert md["frozen_layers"] == sorted(FIRST)
    assert md["method"] == "padding_frozen" and md["freeze_first_layer"] is True


def test_padrl_path_still_trains_every_layer(tmp_path):
    before, execution = _run(tmp_path, freeze=False)
    after = execution.model.policy.state_dict()
    for name in FIRST:
        assert _max_change(before, after, name) > 0.0, name
    assert execution.metadata["freeze_first_layer"] is False
    assert "freeze_check" not in execution.metadata


def _module(model, dotted):
    m = model.policy
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m
