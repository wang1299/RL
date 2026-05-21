"""
Lightweight check for the parallel Habitat scene scheduler.

This does not start Habitat. It only verifies that scene assignment is global:
with 50 scenes and 12 workers, the first 50 assignments visit scene 1..50
exactly once before epoch 2 starts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from RL_training.runner.parallel_habitat_rl_train_runner import ParallelHabitatRLTrainRunner


def _make_runner(scene_count=50, num_workers=12):
    runner = object.__new__(ParallelHabitatRLTrainRunner)
    runner.scene_count = scene_count
    runner.num_workers = num_workers
    runner._reset_scene_scheduler()
    return runner


def test_global_scene_assignment_order():
    runner = _make_runner(scene_count=50, num_workers=12)

    assignments = []
    for i in range(62):
        env_id = i % runner.num_workers
        scene_index = runner._claim_scene_for_env(env_id)
        assignments.append(
            (
                scene_index + 1,
                runner.per_env_scene_epochs[env_id],
                runner.per_env_scene_assignment_numbers[env_id],
            )
        )

    first_epoch = assignments[:50]
    second_epoch_start = assignments[50:62]

    assert [scene for scene, _, _ in first_epoch] == list(range(1, 51))
    assert all(epoch == 1 for _, epoch, _ in first_epoch)
    assert [scene for scene, _, _ in second_epoch_start] == list(range(1, 13))
    assert all(epoch == 2 for _, epoch, _ in second_epoch_start)
    assert [assignment for _, _, assignment in assignments] == list(range(1, 63))


if __name__ == "__main__":
    test_global_scene_assignment_order()
    print("Scene scheduler order OK")
