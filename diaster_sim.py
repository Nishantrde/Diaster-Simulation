import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

print("Backend:", matplotlib.get_backend())

# -----------------------------
# Grid
# -----------------------------
nx, ny = 160, 160
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)

# -----------------------------
# Cyclone center comes from RIGHT and moves LEFT
# -----------------------------
def cyclone_center(t):
    cx = 2.5 - 0.02 * t      # moves from x = +2.5 → -2.5
    cy = 0.6 * np.sin(0.1*t) # slight vertical wobble
    return cx, cy

# -----------------------------
# Safe corners
# -----------------------------
safe_points = np.array([
    [x[0], y[-1]],     # top-left
    # [x[-1], y[-1]],    # top-right
    [x[0], y[0]],      # bottom-left
    # [x[-1], y[0]],     # bottom-right
])

# -----------------------------
# Civilians
# -----------------------------
num_civ = 30
civ_pos = np.random.uniform(-1.0, 1.0, (num_civ, 2))
civ_vel = np.zeros_like(civ_pos)  # store velocity for arrows
arrived = np.zeros(num_civ, dtype=bool)  # track who reached safety

# -----------------------------
# Danger = Gaussian vortex pressure
# -----------------------------
def danger_field(cx, cy):
    R = 3.0   # <-- set your cyclone radius here
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (R**2))

# gradient of scalar
def grad(F):
    Fx = np.gradient(F, axis=1)
    Fy = np.gradient(F, axis=0)
    return Fx, Fy

# -----------------------------
# Curl wind field (vortex)
# -----------------------------
def wind_field(cx, cy):
    Xc = X - cx
    Yc = Y - cy
    r = np.sqrt(Xc**2 + Yc**2) + 1e-9
    strength = 1.0
    u = -strength * (Yc / (r**2))
    v =  strength * (Xc / (r**2))
    return u, v

# -----------------------------
# Move civilian toward nearest safe corner
# -----------------------------
def direction_to_safety(px, py):
    d = np.sqrt((safe_points[:,0]-px)**2 + (safe_points[:,1]-py)**2)
    idx = np.argmin(d)
    tx, ty = safe_points[idx]
    dx = tx - px
    dy = ty - py
    L = np.sqrt(dx*dx + dy*dy) + 1e-9
    return dx/L, dy/L, np.min(d)  # also return distance to nearest safe point

# -----------------------------
# Matplotlib setup
# -----------------------------
fig, ax = plt.subplots(figsize=(6.2,6.2))
ax.set_xticks([]); ax.set_yticks([])

# Initial frame
cx0, cy0 = cyclone_center(0)
D0 = danger_field(cx0, cy0)
im = ax.imshow(D0, extent=(x.min(),x.max(),y.min(),y.max()),
               origin="lower", cmap="inferno", alpha=0.8)

# Quiver (vector arrows for cyclone wind)
stride = 8
U0, V0 = wind_field(cx0, cy0)
quiv = ax.quiver(X[::stride,::stride], Y[::stride,::stride],
                 U0[::stride,::stride], V0[::stride,::stride],
                 color="cyan", alpha=0.8)

# Safe corners
ax.scatter(safe_points[:,0], safe_points[:,1],
           s=120, c="lime", edgecolors="black")

# Civilians
civ_scatter = ax.scatter(civ_pos[:,0], civ_pos[:,1],
                         c="white", s=45, edgecolors="black")

# Quiver for civilians — bigger & more visible
civ_quiv = ax.quiver(civ_pos[:,0], civ_pos[:,1],
                     civ_vel[:,0], civ_vel[:,1],
                     color="white", scale=5, width=0.008)

# -----------------------------
# Animation
# -----------------------------
def update(f):
    global civ_pos, civ_vel, arrived

    t = f * 0.15
    cx, cy = cyclone_center(t)

    # update danger
    D = danger_field(cx, cy)
    im.set_data(D)

    # danger gradient (pushes people away)
    Dx, Dy = grad(D)

    # wind field (curl)
    U, V = wind_field(cx, cy)
    quiv.set_UVC(U[::stride,::stride], V[::stride,::stride])

    # move civilians
    for i in range(num_civ):
        if arrived[i]:
            civ_vel[i] = np.array([0.0, 0.0])
            continue

        px, py = civ_pos[i]
        ix = np.argmin(np.abs(x - px))
        iy = np.argmin(np.abs(y - py))

        # move away from cyclone using gradient
        away = np.array([-Dx[iy,ix], -Dy[iy,ix]])

        # wind pushes them
        wind = np.array([U[iy,ix], V[iy,ix]])

        # move toward safety
        sx, sy, dist_safe = direction_to_safety(px, py)
        safety = np.array([sx, sy]) * 0.4

        # total velocity
        vel = 1.8*away + 0.25*wind + safety
        civ_pos[i] += vel * 0.02
        civ_vel[i] = vel

        # check if reached safe point
        if dist_safe < 0.05:
            arrived[i] = True
            civ_vel[i] = np.array([0.0, 0.0])

    civ_scatter.set_offsets(civ_pos)
    civ_quiv.set_offsets(civ_pos)
    civ_quiv.set_UVC(civ_vel[:,0], civ_vel[:,1])

    ax.set_title(f"Cyclone Evacuation — Cyclone entering from RIGHT → LEFT  t={t:.2f}")
    return im, quiv, civ_scatter, civ_quiv

ani = FuncAnimation(fig, update, frames=400, interval=35, blit=False)

backend = matplotlib.get_backend().lower()
if backend.startswith("agg"):
    ani.save("civilian_vectors_safe.mp4", fps=30, dpi=130)
    print("Saved: civilian_vectors_safe.mp4")
else:
    plt.show()
