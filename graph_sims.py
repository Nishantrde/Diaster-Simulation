# disaster_sim.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import RegularGridInterpolator
import os

# ---------- Parameters (tweak these) ----------
nx, ny = 140, 90            # grid resolution
alpha = 1.0                 # weight of gradient (potential)
beta  = 1.2                 # weight of curl (stream)
n_particles = 360
trail_length = 14
dt = 0.012
frames = 300                # change for longer/shorter animation
# ------------------------------------------------

# domain
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# potential field (phi): global slope + hills/valleys
slope = -1.5
phi = slope * X

def gaussian(x0, y0, sx, sy, amp):
    return amp * np.exp(-(((X-x0)/sx)**2 + ((Y-y0)/sy)**2))

# obstacles / sinks (edit these to match your region)
phi += gaussian(0.35, 0.6, 0.07, 0.07, +1.2)   # hill (repels)
phi += gaussian(0.75, 0.35, 0.12, 0.04, -2.2)  # valley / sink (attracts)
phi += gaussian(0.55, 0.8, 0.03, 0.03, +0.9)
phi += 0.12 * (np.sin(6*X) * np.cos(6*Y))      # small roughness

# stream function (psi): vortices / eddies
psi = np.zeros_like(X)
psi += 0.6 * gaussian(0.4, 0.4, 0.06, 0.06, 1.0)
psi -= 0.9 * gaussian(0.6, 0.5, 0.09, 0.09, 1.0)
psi += 0.5 * gaussian(0.2, 0.75, 0.05, 0.05, 1.0)

# compute fields
dphidy, dphidx = np.gradient(phi, dy, dx)   # returns [d/dy, d/dx]
u_p = dphidx; v_p = dphidy

dpsidy, dpsidx = np.gradient(psi, dy, dx)
u_s = dpsidy; v_s = -dpsidx

U = alpha * u_p + beta * u_s
V = alpha * v_p + beta * v_s

# scale speeds reasonably
speed = np.sqrt(U**2 + V**2)
maxspeed = np.percentile(speed, 98)
U = U / (maxspeed + 1e-8) * 1.0
V = V / (maxspeed + 1e-8) * 1.0

# hazard map (speed + convergence)
dVdy, dUdx = np.gradient(V, dy, dx)[0], np.gradient(U, dy, dx)[1]
divergence = dUdx + dVdy
hazard = np.clip((speed + 0.6 * (-divergence)), 0, None)

# initialize particles (debris/water parcels)
np.random.seed(2)
particles = np.zeros((n_particles, 2))
particles[:,0] = np.random.uniform(0.05, 0.35, size=n_particles)  # x start region
particles[:,1] = np.random.uniform(0.6, 0.95, size=n_particles)   # y start region

trails = -np.ones((n_particles, trail_length, 2))

# interpolators with bounds safety
u_interp = RegularGridInterpolator((y, x), U, bounds_error=False, fill_value=None)
v_interp = RegularGridInterpolator((y, x), V, bounds_error=False, fill_value=None)

def sample_velocity(pos):
    pts = np.stack([pos[:,1], pos[:,0]], axis=-1)
    eps = 1e-9
    pts[:,0] = np.clip(pts[:,0], y[0]+eps, y[-1]-eps)
    pts[:,1] = np.clip(pts[:,1], x[0]+eps, x[-1]-eps)
    u = u_interp(pts)
    v = v_interp(pts)
    return np.stack([u, v], axis=1)

# RK4 integrator (stable)
def rk4_step(pos, dt):
    k1 = sample_velocity(pos)
    k2 = sample_velocity(pos + 0.5*dt*k1)
    k3 = sample_velocity(pos + 0.5*dt*k2)
    k4 = sample_velocity(pos + dt*k3)
    return pos + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def apply_boundaries(pos):
    pos[:,0] = np.clip(pos[:,0], 0.0, 1.0)
    pos[:,1] = np.clip(pos[:,1], 0.0, 1.0)
    return pos

# --- plotting setup ---
fig, ax = plt.subplots(figsize=(9,6))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Disaster-prone area — gradient + curl flow (particles = debris)")
bg = ax.imshow(hazard, origin='lower', extent=(0,1,0,1), cmap='inferno', alpha=0.92, zorder=0)
cb = fig.colorbar(bg, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Hazard index')

step = max(1, nx // 15)   # quiver sampling
ax.quiver(X[::step,::step], Y[::step,::step], U[::step,::step], V[::step,::step], scale=20, width=0.0035, zorder=2, alpha=0.9)
scat = ax.scatter(particles[:,0], particles[:,1], s=9, c='cyan', edgecolors='k', linewidths=0.25, zorder=3)
lines = [ax.plot([], [], linewidth=1, alpha=0.6)[0] for _ in range(n_particles)]

def init():
    scat.set_offsets(particles)
    for ln in lines:
        ln.set_data([], [])
    return [scat] + lines

def update(frame):
    global particles, trails
    newpos = rk4_step(particles, dt)
    newpos += np.random.normal(scale=0.0012, size=newpos.shape)  # small diffusion
    newpos = apply_boundaries(newpos)
    trails = np.roll(trails, shift=1, axis=1)
    trails[:,0,:] = newpos

    # absorb and respawn in sink region (simulate deposition)
    sink_center = np.array([0.75, 0.35])
    dist_to_sink = np.linalg.norm(newpos - sink_center, axis=1)
    absorbed = dist_to_sink < 0.03
    if np.any(absorbed):
        newpos[absorbed,0] = np.random.uniform(0.05, 0.35, size=np.sum(absorbed))
        newpos[absorbed,1] = np.random.uniform(0.6, 0.95, size=np.sum(absorbed))
        trails[absorbed,:,:] = -1

    particles = newpos.copy()
    scat.set_offsets(particles)

    # only draw a subset of trails (speed)
    for i, ln in enumerate(lines):
        if i % 6 == 0:
            valid = trails[i,:,0] >= 0
            if np.sum(valid) > 1:
                ln.set_data(trails[i,valid,0], trails[i,valid,1])
            else:
                ln.set_data([], [])
        else:
            ln.set_data([], [])
    return [scat] + lines

anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=30, blit=False)

# try saving (gif) but heavy; script will always save a preview PNG
out_gif = 'disaster_simulation.gif'
out_mp4 = 'disaster_simulation.mp4'
out_png = 'disaster_simulation_preview.png'
saved = False

# Attempt GIF via Pillow (fallback if ffmpeg not installed)
try:
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=20)
    anim.save(out_gif, writer=writer, dpi=120, savefig_kwargs={'facecolor':fig.get_facecolor()})
    print("Saved GIF to", out_gif)
    saved = True
except Exception as e:
    print("GIF save failed:", e)

# If you prefer MP4 and have ffmpeg: uncomment below (and ensure ffmpeg installed)
# try:
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=25, bitrate=1800)
#     anim.save(out_mp4, writer=writer, dpi=150)
#     print("Saved MP4 to", out_mp4)
#     saved = True
# except Exception as e:
#     print("MP4 save failed:", e)

# Always save a small static preview of the final state
for _ in range(30):
    particles = rk4_step(particles, dt)
    particles = apply_boundaries(particles)

fig2, ax2 = plt.subplots(figsize=(9,6))
ax2.imshow(hazard, origin='lower', extent=(0,1,0,1), cmap='inferno', alpha=0.92)
ax2.quiver(X[::step,::step], Y[::step,::step], U[::step,::step], V[::step,::step], scale=20, width=0.0035)
ax2.scatter(particles[:,0], particles[:,1], s=9, c='cyan', edgecolors='k', linewidths=0.25)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title("Preview (final frame)")
fig2.savefig(out_png, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig); plt.close(fig2)

print("Preview image saved to", out_png)
if not saved:
    print("Animation file not saved (this can take a while). Run locally and uncomment the MP4 block for a faster, high-quality mp4 if you have ffmpeg.")
