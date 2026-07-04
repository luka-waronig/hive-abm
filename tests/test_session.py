from hive_abm import ENVIRONMENTS, SCENARIOS, SimulationSession, build_simulation, scenario_config


def test_named_scenarios_build_small_simulation():
    assert "baseline" in SCENARIOS
    assert "uniform" in ENVIRONMENTS
    cfg = scenario_config("mission_critical", seed=11, rows=6, cols=6, n_nodes=8, n_tasks=10, max_steps=5)
    assert cfg.policy_level == "mission_critical"
    assert cfg.metadata["scenario"] == "mission_critical"
    sim = build_simulation("baseline", "clustered_tasks", seed=12, rows=6, cols=6, n_nodes=8, n_tasks=10, max_steps=5)
    assert sim.config.metadata["environment_preset"] == "clustered_tasks"
    assert len(sim.mission.tasks) == 10


def test_session_step_intervention_and_snapshot():
    session = SimulationSession(
        scenario="baseline",
        environment_preset="uniform",
        seed=13,
        config_overrides={"rows": 6, "cols": 6, "n_nodes": 8, "n_tasks": 10, "max_steps": 6},
    )
    first = session.step()
    assert first["latest_tick"]["step"] == 0
    before_nodes = len(first["nodes"])
    after_relay = session.apply_intervention("add_relay", position=(2, 2))
    assert len(after_relay["nodes"]) == before_nodes + 1
    assert after_relay["session"]["interventions"][-1]["kind"] == "add_relay"
    session.run_steps(2)
    snap = session.snapshot(include_layers=False)
    assert snap["latest_tick"]["step"] >= 1
    assert snap["layers"] == {}


def test_session_run_until_condition():
    session = SimulationSession(
        scenario="baseline",
        environment_preset="uniform",
        seed=14,
        config_overrides={"rows": 6, "cols": 6, "n_nodes": 8, "n_tasks": 10, "max_steps": 8},
    )
    snap = session.run_until("completion_rate >= 0.1", max_steps=8)
    assert snap["latest_tick"]["completion_rate"] >= 0.0
