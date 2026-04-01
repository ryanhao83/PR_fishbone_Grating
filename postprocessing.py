import numpy as np
import matplotlib.pyplot as plt
import json

plot_mode = 'bands'  # 'bands' 或 'fields'

if plot_mode == 'bands':
    # Load npz
    # npz = np.load('cache_2d/fishbone2d_hspine0p380_res32_nb19_974a1f09.npz', allow_pickle=True)
    # npz = np.load('cache_2d/fishbone2d_hspine0p400_res32_nb19_1dddb44b.npz', allow_pickle=True)
    npz = np.load('cache_2d/fishbone2d_hspine0p420_res32_nb19_3ae4efdf.npz', allow_pickle=True)

    k       = npz['k_x']                # k 点
    freqs   = npz['freqs_even']          # shape (nk, nb)，所有能带频率
    nb      = freqs.shape[1]             # 能带数
    info    = json.loads(str(npz['params_json']))
    a_nm    = info['a_nm']               # 晶格常数 (nm)
    wl_nm   = a_nm / freqs               # shape (nk, nb)，对应波长 (nm)

    # 光锥: freq = k / n_clad
    n_clad = info.get('n_SiO2', 1.44)
    light_line = k / n_clad

    # 画全部能带
    fig, (ax, ax0, ax1) = plt.subplots(1, 3, figsize=(18, 5))
    ax.fill_between(k, light_line, light_line.max() * 1.2,
                     color='lightyellow', alpha=0.6, label='light cone')
    ax.plot(k, light_line, 'k--', lw=1.0, label=f'light line (n={n_clad})')
    for b in range(nb):
        ax.plot(k, freqs[:, b], lw=0.8, alpha=0.7, label=f'band {b}')
    ax.set_ylim(0.15, freqs.max() * 1.05)
    ax.set_xlabel('k (2π/a)')
    ax.set_ylabel('freq (a/λ)')
    ax.set_title(f"h_spine={info['h_spine']:.2f}a — all {nb} bands")
  
    # 只看某一条带 (例如 band 11)
    b_group = [10, 15, 16]
    for b in b_group:
        ax0.plot(k, freqs[:, b], label=f'band {b}')
    ax0.set_xlabel('k (2π/a)') 
    ax0.set_ylabel('freq (a/λ)')
    ax0.set_title(f"band {b} dispersion")
    ax0.grid()
    ax0.legend()


    # vg/c = dfreq/dk (MPB normalized units), ng = c/vg = 1/(dfreq/dk)
    for b in b_group:
        vg = np.gradient(freqs[:, b], k)   # dω/dk in units of c
        ng = -1.0 / vg                      # group index ng = c/vg
        ax1.plot(wl_nm[:, b], ng, label=f'band {b}')
    ax1.set_xlabel('wavelength λ (nm)')
    ax1.set_ylabel('$n_g$')
    ax1.set_title(f"band {b} group index")
    ax1.set_ylim(0, 90)
    ax1.grid()
      
    ax1.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)



elif plot_mode == 'fields':

    # 加载模场文件
    data = dict(np.load('cache_2d/fields_hspine0p400_k0p480_nb18.npz', allow_pickle=True))

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

elif plot_mode == 'ng':
    # 画色散关系，计算群速度
    data = np.loadtxt('cache_2d/bands_hspine0p400_nb18.csv', delimiter=',', skiprows=1)
    k       = data[:, 0]            # k 点
    nb      = 18                    # 能带数
    freqs   = data[:, 1:1+nb]       # shape (nk, nb)，所有能带频率

    b = 11  # 选择某一条带
    v_g = np.gradient(freqs[:, b], k)  # 群速度 v_g = dω/dk
    plt.plot(k, v_g)
    plt.xlabel('k (2π/a)')
    plt.ylabel('group velocity v_g (a/λ)')
    plt.title(f'Group velocity of band {b}')
    plt.grid()
    plt.show()