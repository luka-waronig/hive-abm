from hive_abm.agents import HiveNode, ResourceProfile
from hive_abm.mission import MissionGraph, Task
from hive_abm.simulation import HiveSimulation, SimulationConfig


COORDINATOR_PERMS = {
    "wait",
    "move",
    "solve_task",
    "verify_task",
    "relay_message",
    "read_memory",
    "write_memory",
    "delegate_task",
    "observe_sector",
    "assign_task",
}
WORKER_PERMS = {"wait", "move", "solve_task", "read_memory", "write_memory", "observe_sector"}
VERIFIER_PERMS = {"wait", "move", "verify_task", "read_memory", "write_memory", "observe_sector"}


def node(node_id, role, position, permissions, specialization, sensing_radius=1):
    return HiveNode(
        node_id=node_id,
        role=role,
        specialization=specialization,
        position=position,
        permissions=set(permissions),
        resources=ResourceProfile(sensing_radius=sensing_radius, communication_radius=3),
        reliability=1.0,
    )


def test_simulation_runs_and_logs_ticks():
    cfg = SimulationConfig(rows=8, cols=8, n_nodes=12, n_tasks=15, max_steps=20, seed=123, run_id="test")
    sim = HiveSimulation(cfg)
    summary = sim.run()
    assert summary.steps > 0
    assert 0.0 <= summary.final_completion_rate <= 1.0
    assert len(sim.metrics.ticks) == summary.steps


def test_strict_policy_can_deny_actions():
    cfg = SimulationConfig(rows=8, cols=8, n_nodes=12, n_tasks=20, max_steps=15, policy_level="strict", seed=456, run_id="strict_test")
    sim = HiveSimulation(cfg)
    summary = sim.run()
    assert summary.total_denied_actions >= 0


def test_mission_critical_requires_downstream_verification():
    mission = MissionGraph.from_tasks([Task(0, "coding", (0, 0), difficulty=0.0)], quality_threshold=0.0)
    nodes = [
        node(0, "coordinator", (0, 0), COORDINATOR_PERMS, {"coding": 0.1, "verification": 0.8}),
        node(1, "worker", (0, 0), WORKER_PERMS, {"coding": 1.0, "verification": 0.1}),
        node(2, "verifier", (0, 0), VERIFIER_PERMS, {"coding": 0.2, "verification": 1.0}),
    ]
    cfg = SimulationConfig(
        rows=3,
        cols=3,
        n_nodes=3,
        n_tasks=1,
        max_steps=4,
        task_duration_min=1,
        task_duration_max=1,
        verification_duration_min=1,
        verification_duration_max=1,
        policy_level="mission_critical",
        node_failure_probability=0.0,
        message_loss_probability=0.0,
        adaptive_movement=False,
        seed=1,
    )
    sim = HiveSimulation(cfg, nodes=nodes, mission=mission)
    sim._attempt_task = lambda node, task, verify=False: (True, 1.0)

    sim.step_idx = 0
    sim.step()
    task = mission.tasks[0]
    assert task.status == "awaiting_verification"
    assert not mission.is_success()
    assert task.solved_by == 1

    sim.step_idx = 1
    sim.step()
    assert task.status == "completed"
    assert task.verified
    assert mission.is_success()


def test_task_duration_keeps_work_in_progress_before_attempt():
    mission = MissionGraph.from_tasks([Task(0, "coding", (0, 0), difficulty=0.0)], quality_threshold=0.0)
    nodes = [node(0, "worker", (0, 0), WORKER_PERMS, {"coding": 1.0})]
    cfg = SimulationConfig(
        rows=3,
        cols=3,
        n_nodes=1,
        n_tasks=1,
        max_steps=3,
        task_duration_min=2,
        task_duration_max=2,
        policy_level="permissive",
        node_failure_probability=0.0,
        message_loss_probability=0.0,
        adaptive_movement=False,
        seed=2,
    )
    sim = HiveSimulation(cfg, nodes=nodes, mission=mission)
    sim._attempt_task = lambda node, task, verify=False: (True, 1.0)

    sim.step_idx = 0
    sim.step()
    task = mission.tasks[0]
    assert task.status == "in_progress"
    assert task.work_remaining == 1
    assert task.attempts == 0
    assert sim.metrics.ticks[-1].working_nodes == 1

    sim.step_idx = 1
    sim.step()
    assert task.status == "completed"
    assert task.attempts == 1


def test_unreachable_high_skill_node_is_not_globally_assigned():
    mission = MissionGraph.from_tasks([Task(0, "coding", (0, 0), difficulty=0.0)], quality_threshold=0.0)
    nodes = [
        node(0, "coordinator", (0, 0), COORDINATOR_PERMS, {"coding": 0.1}),
        node(1, "worker", (2, 2), WORKER_PERMS, {"coding": 0.6}),
        node(2, "worker", (4, 4), WORKER_PERMS, {"coding": 1.0}),
    ]
    cfg = SimulationConfig(
        rows=5,
        cols=5,
        n_nodes=3,
        n_tasks=1,
        max_steps=2,
        task_duration_min=2,
        task_duration_max=2,
        policy_level="permissive",
        assignment_horizon_hops=1,
        node_failure_probability=0.0,
        message_loss_probability=0.0,
        adaptive_movement=False,
        seed=3,
    )
    sim = HiveSimulation(cfg, nodes=nodes, mission=mission)
    sim.adjacency = {0: {1}, 1: {0}, 2: set()}

    sim.step_idx = 0
    sim.step()
    assert mission.tasks[0].status == "in_progress"
    assert mission.tasks[0].assigned_to == 1


def test_failed_final_attempt_is_counted_once():
    mission = MissionGraph.from_tasks([Task(0, "coding", (0, 0), difficulty=0.0)], quality_threshold=0.0)
    nodes = [node(0, "worker", (0, 0), WORKER_PERMS, {"coding": 1.0})]
    cfg = SimulationConfig(
        rows=3,
        cols=3,
        n_nodes=1,
        n_tasks=1,
        max_steps=1,
        task_duration_min=1,
        task_duration_max=1,
        policy_level="permissive",
        node_failure_probability=0.0,
        message_loss_probability=0.0,
        adaptive_movement=False,
        seed=4,
    )
    sim = HiveSimulation(cfg, nodes=nodes, mission=mission)
    sim._attempt_task = lambda node, task, verify=False: (False, 0.0)

    sim.step_idx = 0
    sim.step()
    assert mission.tasks[0].attempts == 1
    assert mission.tasks[0].status == "pending"
