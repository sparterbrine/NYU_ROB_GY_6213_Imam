from time import time

# Predefined trajectories.
# Each trajectory is a list of (speed, steering_angle, duration_seconds) triples.
# Speed: 0–100, steering_angle: -20–20 (degrees), duration: seconds.
TRAJECTORIES = {
    "Straight Line": [
        (100, 0, 3.0),
    ],
    "Square": [
        (100,  0, 2.0),
        (100, 15, 1.5),
        (100,  0, 2.0),
        (100, 15, 1.5),
        (100,  0, 2.0),
        (100, 15, 1.5),
        (100,  0, 2.0),
        (100, 15, 1.5),
    ],
    "S-Curve": [
        (100,  10, 3.0),
        (100, -10, 3.0),
        (100,   0, 3.0),
    ],
}


class TrajectoryRunner:
    """Executes a predefined trajectory segment by segment.

    Call start(name) to begin, stop() to abort, and update() once per
    control-loop tick to get the current (speed, steering_angle) command.
    """

    def __init__(self):
        self._trajectory = None
        self._segment_index = 0
        self._segment_start_time = 0.0
        self.is_running = False

    def start(self, trajectory_name: str):
        """Start running the named trajectory from the first segment."""
        if trajectory_name not in TRAJECTORIES:
            raise ValueError(f"Unknown trajectory: '{trajectory_name}'")
        self._trajectory = TRAJECTORIES[trajectory_name]
        self._segment_index = 0
        self._segment_start_time = time()
        self.is_running = True

    def stop(self):
        """Abort the currently running trajectory."""
        self.is_running = False
        self._trajectory = None

    def update(self) -> tuple[int, int]:
        """Advance the trajectory and return (speed, steering_angle).

        Moves to the next segment when the current one's duration has elapsed.
        Returns (0, 0) and sets is_running=False when all segments are done.
        """
        if not self.is_running or self._trajectory is None:
            return 0, 0

        now = time()
        speed, steering, duration = self._trajectory[self._segment_index]

        if now - self._segment_start_time >= duration:
            self._segment_index += 1
            if self._segment_index >= len(self._trajectory):
                self.is_running = False
                return 0, 0
            self._segment_start_time = now
            speed, steering, _ = self._trajectory[self._segment_index]

        return speed, steering
