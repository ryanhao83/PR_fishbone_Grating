import numpy as np
import json

# Load CSV
data = np.loadtxt('output_2d/bands_hspine0p400_nb18.csv', delimiter=',', skiprows=1)

k       = data[:, 0]            # k 点
nb      = 18                    # 能带数
freqs   = data[:, 1:1+nb]       # shape (nk, nb)，所有能带频率
wl_nm   = data[:, 1+nb:1+2*nb] # shape (nk, nb)，对应波长 (nm)

# 读参数信息
with open('output_2d/bands_hspine0p400_nb18_info.json') as f:
    info = json.load(f)

# 画全部能带
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for b in range(nb):
    ax.plot(k, freqs[:, b], lw=0.8, alpha=0.7, label=f'band {b}')
ax.set_xlabel('k (2π/a)')
ax.set_ylabel('freq (a/λ)')
ax.set_title(f"h_spine={info['params']['h_spine']:.2f}a — all {nb} bands")
plt.tight_layout()
plt.show()

# 只看某一条带 (例如 band 11)
b = 11
plt.plot(wl_nm[:, b], np.gradient(1/freqs[:, b], k))  # 只是例子



import numpy as np
import matplotlib.pyplot as plt

# 加载模场文件
data = dict(np.load('output_2d/fields_hspine0p400_k0p480_nb18.npz', allow_pickle=True))

eps   = data['epsilon']   # (nx, ny) 介电常数
freqs = data['freqs']     # (nb,) 各 band 在该 k 点的频率
k_at  = float(data['k_at'])
nb    = int(data['num_bands'])

print(f"k = {k_at},  num_bands = {nb}")
for b in range(nb):
    print(f"  band {b:2d}: freq = {freqs[b]:.6f}  (λ ≈ {420/freqs[b]:.1f} nm)")

# 取某一个 band 的场 (例如 band 11)
b = 11
E = data[f'efield_b{b}']   # (nx, ny, 3) 复数，分量顺序 Ex, Ey, Ez
H = data[f'hfield_b{b}']   # (nx, ny, 3) 复数，分量顺序 Hx, Hy, Hz

sy = float(data['sy'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# |Ey|
axes[0].imshow(np.abs(E[:, :, 1]).T, origin='lower', cmap='hot',
               extent=[-0.5, 0.5, -sy/2, sy/2], aspect='auto')
axes[0].set_title(f'|Ey| band {b}')
axes[0].set_xlabel('x (a)'); axes[0].set_ylabel('y (a)')
plt.colorbar(axes[0].images[0], ax=axes[0])

# |Ex|
axes[1].imshow(np.abs(E[:, :, 0]).T, origin='lower', cmap='hot',
               extent=[-0.5, 0.5, -sy/2, sy/2], aspect='auto')
axes[1].set_title(f'|Ex| band {b}')
axes[1].set_xlabel('x (a)')
plt.colorbar(axes[1].images[0], ax=axes[1])

# 介电常数 (结构背景)
axes[2].imshow(eps.T, origin='lower', cmap='gray',
               extent=[-0.5, 0.5, -sy/2, sy/2], aspect='auto')
axes[2].set_title('ε (geometry)')
axes[2].set_xlabel('x (a)')
plt.colorbar(axes[2].images[0], ax=axes[2])

plt.suptitle(f'h_spine=0.40a, k={k_at}, freq={freqs[b]:.6f}')
plt.tight_layout()
plt.savefig(f'field_band{b}_k{k_at:.2f}.png', dpi=200)
plt.show()