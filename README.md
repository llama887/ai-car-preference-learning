# Shapes

## Segment Collection

**Segments** are lists of `StateActionPair` objects. A segment of length `k` has length `(k+1)`. Note that a `Segment` is an actual object we have programmatically defined.

When a segment of length `k` is flattened into a list (via `prepare_single_trajectory`), the resulting list has a size of `((size of StateActionPair) * (k+1))`.


- **`database_gargantuar_k_length.pkl`**
  A database storing segments of length `k`. The structure is a `list[list[Segment]]`, where the outer list's first index specifies the number of rules satisfied. For example:
  - `data[i]` contains a list of segments of length `k` that satisfy `i` rules.

- **`database_x_pairs_y_rules_z_length`**
  A sampled database stored as a list of 5-tuples with the following structure:
  - `segment_1: Segment`
  - `segment_2: Segment`
  - `label: int` (given by `reward_1 < reward_2`)
  - `reward_1: int`
  - `reward_2: int`

---

## Trajectory Storage

A **Trajectory** is an object that contains the following members:

- **`traj`**: `list[Segment]`
  Contains the trajectory (a sequence of segments).

- **`num_expert_segments`**: `int`
  The number of segments within the trajectory that satisfy all rules.

- **`reward`**: `float`
  The total reward accumulated across all segments in the trajectory.
  - Note: This value equals `num_expert_segments` when using a ground truth reward function that assigns a reward of `1` to a segment satisfying all rules, and `0` otherwise.

### Example Databases

- **`trueRF_x_trajectories.pkl`**
  A list of `x` trajectories produced by ground truth agents. The structure is:
  - `list[Trajectory]`

- **`trainedRF_x_trajectories.pkl`**
  A list of `x` trajectories produced by agents rewarded under a trained reward network. The structure is:
  - `list[Trajectory]`
