# Cyclone Evacuation Simulation

This project simulates civilians evacuating a region threatened by a moving cyclone. It visualizes the interaction between a moving danger zone, a rotational wind field, and autonomous agents (civilians) attempting to reach safe zones.

## Files

- `diaster_sim.py`: The main simulation script (Cyclone Evacuation).
- `graph_sims.py`: A secondary simulation showing particle advection in a disaster-prone area with terrain features.

## Installation

Requires Python 3 and the following libraries:

```bash
pip install numpy matplotlib scipy
```

## Usage

Run the simulation script:

```bash
python diaster_sim.py
```

- **Interactive Mode:** If run on a machine with a display, a window will open showing the animation.
- **Headless Mode:** If run on a server or without a display, it will automatically save the animation as `civilian_vectors_safe.mp4`.

## Code Documentation (`diaster_sim.py`)

### Global Variables & Grid Setup

- **`nx, ny`**: (160, 160) Resolution of the simulation grid.
- **`x, y`**: Linear arrays defining the coordinate space from -2 to 2.
- **`X, Y`**: 2D Meshgrids representing the full coordinate plane. These are used to calculate fields (danger, wind) across the entire map simultaneously.
- **`safe_points`**: A list of coordinates `[x, y]` representing "Safe Zones" (green dots). Civilians aim for these points.
- **`num_civ`**: (30) The number of civilians in the simulation.
- **`civ_pos`**: A `(30, 2)` NumPy array storing the current `x` and `y` coordinates of every civilian.
- **`civ_vel`**: Stores the current velocity vector for each civilian (used for visualization).
- **`arrived`**: A boolean array tracking which civilians have successfully reached a safe zone.

### Functions

#### `cyclone_center(t)`
Calculates the current position of the cyclone's eye.
- **Input**: Time `t`.
- **Logic**: Moves the center linearly from Right (`x=2.5`) to Left, while adding a sinusoidal vertical "wobble".
- **Returns**: `(cx, cy)` coordinates.

#### `danger_field(cx, cy)`
Computes the "Danger Level" (scalar field) across the entire grid.
- **Logic**: Uses a Gaussian function $\exp(-r^2/R^2)$ centered at the cyclone.
- **Purpose**: Used to visualize the heat map and to calculate the gradient (direction of steepest descent) so civilians can run *away* from the peak danger.

#### `grad(F)`
A helper function to calculate the gradient of any 2D field `F`.
- **Returns**: `Fx` (change in x) and `Fy` (change in y).
- **Usage**: `Dx, Dy = grad(D)` gives the vector direction pointing *towards* higher danger. Civilians move in the opposite direction (`-Dx, -Dy`).

#### `wind_field(cx, cy)`
Computes the wind velocity vector field.
- **Logic**: Generates a rotational vortex around the cyclone center.
  - $u \propto -Y/r^2$
  - $v \propto X/r^2$
- **Returns**: `u` (x-velocity grid) and `v` (y-velocity grid).

#### `direction_to_safety(px, py)`
Calculates the vector pointing to the nearest safe zone for a specific civilian.
- **Input**: Civilian position `(px, py)`.
- **Logic**: Finds the closest point in `safe_points`, calculates the vector difference, and normalizes it (makes length = 1).
- **Returns**: Unit vector components `(dx, dy)` and the distance to the safe point.

#### `update(f)`
The main loop called by `FuncAnimation` for every frame `f`.
1.  **Environment**: Updates cyclone position `(cx, cy)`, danger field, and wind field.
2.  **Agents**: Iterates through each civilian:
    -   **Gradient Force**: Calculates vector away from danger (`-grad(Danger)`).
    -   **Wind Force**: Samples the wind vector at the civilian's location.
    -   **Goal Force**: Calculates vector towards the nearest safe zone.
    -   **Integration**: Combines forces to update position:
        $$ \vec{v}_{new} = 1.8 \cdot \vec{v}_{escape} + 0.25 \cdot \vec{v}_{wind} + 0.4 \cdot \vec{v}_{safety} $$
    -   **Arrival Check**: If distance to safety < 0.05, stops the civilian.
3.  **Drawing**: Updates the positions of the scatter plot (dots) and quiver plot (arrows).
