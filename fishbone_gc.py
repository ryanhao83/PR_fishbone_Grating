import meep as mp
import meep.mpb as mpb
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义几何与材料参数
# ==========================================
# 晶格常数 a (周期)。在 MPB 中，所有空间尺度都以 a 为单位。
# 根据图3(b)的波长范围(1550~1630nm)和图3(a)的归一化频率(~0.262)
# 估算周期 a 约为 420 nm。
a_nm = 420.0 

# 从文献图中提取的几何参数 (单位: a)
w1 = 0.52
w2 = 0.80
w3 = 0.68

h1 = 1.70
h2 = 0.85
h3 = 1.75

# [关键缺失参数]：中心连续脊波导的半宽度
# 文献未直接给出，这里假设一个合理的值，您可以根据实际情况修改
h_spine = 0.4 

# 波导厚度 (假设标准的 220nm SOI 硅波导)
t_slab = 220.0 / a_nm  

# 超原胞 (Supercell) 尺寸
# Y方向(横向)需要包容所有结构并留出足够的真空衰减区
sy = 2 * (h_spine + h1 + h2 + h3) + 4.0 
# Z方向(垂直)留出足够的包层区
sz = t_slab + 4.0 

geometry_lattice = mp.Lattice(size=mp.Vector3(1, sy, sz))

# 硅 (Si) 和 二氧化硅 (SiO2) 在 1550nm 附近的近似折射率
Si = mp.Medium(index=3.48)
SiO2 = mp.Medium(index=1.44)

# ==========================================
# 2. 构建几何结构 (Geometry)
# ==========================================
geometry = [
    # SiO2 衬底 (占据 Z < -t_slab/2 的空间)
    mp.Block(size=mp.Vector3(mp.inf, mp.inf, sz/2),
             center=mp.Vector3(0, 0, -t_slab/2 - sz/4),
             material=SiO2),
             
    # 中心连续脊波导
    mp.Block(size=mp.Vector3(mp.inf, 2 * h_spine, t_slab),
             center=mp.Vector3(0, 0, 0),
             material=Si)
]

# 添加两侧的周期性肋条 (Ribs)
# 上半部分 (Y > 0)
geometry.append(mp.Block(size=mp.Vector3(w3, h3, t_slab), center=mp.Vector3(0, h_spine + h3/2, 0), material=Si))
geometry.append(mp.Block(size=mp.Vector3(w2, h2, t_slab), center=mp.Vector3(0, h_spine + h3 + h2/2, 0), material=Si))
geometry.append(mp.Block(size=mp.Vector3(w1, h1, t_slab), center=mp.Vector3(0, h_spine + h3 + h2 + h1/2, 0), material=Si))

# 下半部分 (Y < 0)，保持对称
geometry.append(mp.Block(size=mp.Vector3(w3, h3, t_slab), center=mp.Vector3(0, -(h_spine + h3/2), 0), material=Si))
geometry.append(mp.Block(size=mp.Vector3(w2, h2, t_slab), center=mp.Vector3(0, -(h_spine + h3 + h2/2), 0), material=Si))
geometry.append(mp.Block(size=mp.Vector3(w1, h1, t_slab), center=mp.Vector3(0, -(h_spine + h3 + h2 + h1/2), 0), material=Si))

# ==========================================
# 3. 仿真设置与运行
# ==========================================
# 设置布里渊区积分路径 (对应图3a的 x 轴范围: 0.36 到 0.5)
# mp.interpolate 会在两点之间插入 50 个点
k_points = mp.interpolate(50, [mp.Vector3(0.36, 0, 0), mp.Vector3(0.5, 0, 0)])

# 这种极宽的结构支持大量模式，带隙或平缓的慢光模式可能不在基模
# 建议先计算前 5 个模式，之后根据电场分布挑选匹配的模式
num_bands = 5 
resolution = 16 # 空间分辨率，若要发表级别精度可调高至 32 或更高

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands
)

print("Starting MPB Simulation...")
ms.run()

# ==========================================
# 4. 数据提取与绘图
# ==========================================
# 提取能带数据
freqs = ms.all_freqs
k_x = np.array([k.x for k in k_points])

# 【注意】您需要在此处指定您要观察的模式阶数（0 为基模）
# 由于结构较宽，慢光模式可能对应较高阶的横向模式，请尝试调整此变量
target_band = 0 
f_band = freqs[:, target_band]

# 计算群折射率 ng
# ng = c / vg = 1 / (df/dk)，其中 f 和 k 都是 MPB 的归一化单位
df_dk = np.gradient(f_band, k_x)
ng = 1.0 / df_dk

# 计算对应的物理波长 (nm)
wavelengths = a_nm / f_band

# 开始绘制图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- 图 (a): 能带图 (Band Diagram) ---
ax1.plot(k_x, f_band, 'r.-', label=f'Band {target_band}')
ax1.set_xlabel('Wave vector ($2\pi/a$)')
ax1.set_ylabel('Normalized frequency ($a/\lambda$)')
ax1.set_title('(a) Band Diagram')
ax1.set_xlim([0.36, 0.5])

# 绘制二氧化硅的光锥 (f = k_x / 1.44)
# 填充光线以上的区域
light_line = k_x / 1.44
ax1.fill_between(k_x, light_line, 1.0, color='gray', alpha=0.3, label='Light Cone')

# 强制限制 Y 轴的显示范围，使其与原论文图 3a 一致
ax1.set_ylim([0.25, 0.28]) 
ax1.grid(True)
ax1.legend()

# --- 图 (b): 群折射率图 (Group Index ng) ---
# 在布里渊区边缘(k=0.5)附近，斜率接近 0，ng 会急剧发散到无穷大
# 我们过滤掉异常大或由于数值波动产生的负值点
valid_idx = (ng > 0) & (ng < 160) 

ax2.plot(wavelengths[valid_idx], ng[valid_idx], 'ro-')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Group index $n_g$')
ax2.set_title('(b) Group Index vs Wavelength')
ax2.set_xlim([1550, 1630])
ax2.set_ylim([0, 160])
ax2.grid(True)

plt.tight_layout()
plt.show()