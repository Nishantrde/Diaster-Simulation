# Cyclone Evacuation Simulation — Presentation Guide

Here's everything you need to explain the simulation clearly and confidently.

---

## **1. THE HEAT MAP (Danger Field)**

### **What It Shows**
The red/yellow colored background represents the **intensity of the cyclone's danger**. Bright yellow = deadly center, dark = safe.

### **How We Created It**
We used a **Gaussian Function** (Bell Curve) centered on the cyclone:

$$D(x,y) = e^{-\frac{(x-c_x)^2 + (y-c_y)^2}{R^2}}$$

```python
def danger_field(cx, cy):
    R = 3.0   # Radius of the storm
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (R**2))
```

### **Key Points to Mention**
| Location | Value | Color |
|----------|-------|-------|
| At cyclone center | 1.0 | Bright Yellow |
| At distance R | 0.37 | Orange |
| Far away | ~0.0 | Black (Safe) |

### **Why Gaussian?**
- **Smooth decay**: No sudden jumps in danger
- **Realistic**: Real storms have gradual falloff
- **Symmetric**: Same danger in all directions from center

---

## **2. THE VECTOR ARROWS (Wind Field)**

### **What They Show**
The **cyan arrows** represent the **wind direction and speed** at each point. They form a **spinning vortex** (counter-clockwise rotation).

### **How We Created It**
We used a **Point Vortex Model** from fluid dynamics:

$$u = -\frac{Y_c}{r^2}, \quad v = \frac{X_c}{r^2}$$

**Where:**
- **$u$** = Horizontal wind velocity (Left/Right)
- **$v$** = Vertical wind velocity (Up/Down)
- **$X_c$** = Horizontal distance from cyclone center (`X - cx`)
- **$Y_c$** = Vertical distance from cyclone center (`Y - cy`)
- **$r$** = Distance from cyclone center = $\sqrt{X_c^2 + Y_c^2}$

```python
def wind_field(cx, cy):
    Xc = X - cx                          # Horizontal distance to center
    Yc = Y - cy                          # Vertical distance to center
    r = np.sqrt(Xc**2 + Yc**2) + 1e-9   # Distance from center
    strength = 1.0
    u = -strength * (Yc / (r**2))        # Horizontal wind
    v =  strength * (Xc / (r**2))        # Vertical wind
    return u, v
```

### **Why Arrow Size Changes**
The arrows get **bigger near the center** and **smaller far away** because:

$$\text{Wind Speed} = \frac{\text{strength}}{r}$$

| Distance from Center | Arrow Size | Wind Speed |
|----------------------|------------|------------|
| Close (r = 0.5) | **Large** | Strong |
| Medium (r = 1.0) | Medium | Medium |
| Far (r = 3.0) | **Small** | Weak |

### **Why Counter-Clockwise?**
The math trick: **Swap X and Y, negate one**.
- `u = -Yc` (use vertical distance for horizontal wind)
- `v = +Xc` (use horizontal distance for vertical wind)

This creates perpendicular (90°) rotation at every point.

---

## **3. THE CYCLONE BEHAVIOR**

### **Movement Pattern**
The cyclone enters from the **RIGHT** and moves to the **LEFT** while **wobbling up and down**.

```python
def cyclone_center(t):
    cx = 2.5 - 0.02 * t       # Linear motion: RIGHT → LEFT
    cy = 0.6 * np.sin(0.1*t)  # Sinusoidal wobble: UP ↔ DOWN
    return cx, cy
```

### **Path Visualization**
```
        y (vertical)
        |
   0.6  |     /\     /\      ← Wobbles UP
        |    /  \   /  \
    0   |---/----\-/----\--- ← Center line
        |  /      X      \
  -0.6  | /        \      \  ← Wobbles DOWN
        |/__________\______\_____ x (horizontal)
       2.5         0        -2.5
     (START)    (CENTER)    (END)
```

### **Key Parameters**
| Parameter | Value | Effect |
|-----------|-------|--------|
| Starting position | `2.5` | Cyclone starts off-screen (right) |
| Speed | `0.02` | How fast it moves left per frame |
| Wobble amplitude | `0.6` | How much it moves up/down |
| Wobble frequency | `0.1` | How fast it wobbles |

---

## **4. THE CIVILIAN PATHFINDING ALGORITHM**

### **Algorithm Name**
**Gradient Descent with Multi-Force Navigation**

### **The Core Equation**
Each civilian's velocity is calculated by combining **three forces**:

$$\vec{V}_{total} = 1.8 \times \vec{V}_{away} + 0.25 \times \vec{V}_{wind} + 0.4 \times \vec{V}_{safety}$$

```python
vel = 1.8*away + 0.25*wind + safety
```

### **The Three Forces Explained**

#### **Force 1: Run Away from Danger (Gradient Descent)**
```python
Dx, Dy = grad(D)                          # Calculate slope of danger
away = np.array([-Dx[iy,ix], -Dy[iy,ix]]) # Negative = run DOWNHILL
```

| What | How |
|------|-----|
| **Gradient** | Points toward danger (uphill) |
| **Negative Gradient** | Points away from danger (downhill) |
| **Weight: 1.8** | Strongest force (survival instinct) |

#### **Force 2: Pushed by Wind (Physics)**
```python
U, V = wind_field(cx, cy)#recalculate wind field centered on new cyclone position
wind = np.array([U[iy,ix], V[iy,ix]])
```

| What | How |
|------|-----|
| **Wind Vector** | Sampled at civilian's position |
| **Effect** | Pushes them sideways (vortex spin) |
| **Weight: 0.25** | Weak force (can resist wind somewhat) |

#### **Force 3: Move Toward Safety (Goal-Seeking)**
```python
sx, sy, dist_safe = direction_to_safety(px, py)
safety = np.array([sx, sy]) * 0.4
```

| What | How |
|------|-----|
| **Direction** | Unit vector pointing to nearest safe zone |
| **Distance Check** | Finds closest of all safe points |
| **Weight: 0.4** | Moderate force (steady pull to goal) |

### **Why These Weights?**

| Force | Weight | Reason |
|-------|--------|--------|
| **Away** | 1.8 | Survival is most important — run from danger first! |
| **Wind** | 0.25 | Can't fully resist wind, but not totally helpless |
| **Safety** | 0.4 | Steady goal, but not as urgent as escaping danger |

---

## **5. VISUALIZATION COMPONENTS**

### **What's Drawn on Screen**

| Element | Code | Description |
|---------|------|-------------|
| **Heat Map** | `ax.imshow(D, cmap='inferno')` | Danger intensity (red/yellow) |
| **Wind Arrows** | `ax.quiver(X, Y, U, V, color='cyan')` | Vortex wind direction |
| **Safe Zones** | `ax.scatter(..., c='lime')` | Green dots (evacuation points) |
| **Civilians** | `ax.scatter(..., c='white')` | White dots (people) |
| **Civilian Velocity** | `ax.quiver(..., color='white')` | Small arrows showing movement |

### **Animation Loop**
Every frame (35ms):
1. Move the cyclone
2. Recalculate danger field
3. Recalculate wind field
4. For each civilian:
   - Calculate three forces
   - Combine into velocity
   - Update position
   - Check if reached safety
5. Redraw everything

---

## **6. QUICK SUMMARY FOR PRESENTATION**

### **One-Liner for Each Component**

| Component | One-Line Explanation |
|-----------|----------------------|
| **Heat Map** | Gaussian bell curve showing danger intensity around the cyclone. |
| **Wind Arrows** | Point vortex model creating counter-clockwise spinning wind. |
| **Arrow Size** | Inversely proportional to distance (1/r) — stronger near center. |
| **Cyclone Path** | Linear left movement + sinusoidal up-down wobble. |
| **Pathfinding** | Gradient Descent: civilians "surf downhill" on the danger map while being pushed by wind and pulled toward safety. |

### **Algorithm Summary**
> "Civilians use **Gradient Descent** to run away from the peak danger, while also being physically **pushed by the vortex wind** and **attracted toward safe zones**. The combination of these three forces creates realistic evacuation behavior."

---

## **7. POSSIBLE QUESTIONS & ANSWERS**

**Q: Why Gaussian for danger?**
> A: It's smooth, symmetric, and realistic — danger fades gradually like a real storm.

**Q: Why do arrows spin counter-clockwise?**
> A: We swap X and Y coordinates and negate one to create perpendicular (tangential) velocity at each point.

**Q: Why not just run straight to safety?**
> A: Real evacuations face obstacles like wind and danger zones. Our algorithm balances survival (avoid danger) with goal-seeking (reach safety).

**Q: What is Gradient Descent?**
> A: It's an optimization algorithm that finds the "downhill" direction. Civilians use it to run away from the danger peak.

**Q: Can civilians die?**
> A: In this simulation, no — but you could add that by checking if danger level exceeds a threshold.

---

## **8. INSTALLATION & USAGE**

### **Requirements**
```bash
pip install numpy matplotlib scipy
```

### **Run the Simulation**
```bash
python diaster_sim.py
```

- **Interactive Mode:** Opens a window showing the animation.
- **Headless Mode:** Saves as `civilian_vectors_safe.mp4`.

### **Generate GIF**
```bash
python diaster_sim_gif.py
```
- Saves as `cyclone_evacuation.gif`.

---

Good luck with your presentation! 🎯🌪️
