import time
import gc
import tracemalloc
import numpy as np
import pandas as pd
import scipy.special
import scipy.fft
from scipy.spatial.distance import cdist
import finufft
import seaborn as sns
import matplotlib.pyplot as plt
from pyscf import gto

from grid import GaussLegendre, BeckeRTransform, BeckeWeights, MolGrid
from grid.angular import AngularGrid


# ==========================================
# 1. Baseline & Initializers
# ==========================================
def get_analytical_baseline():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp')
    eri_analytical = mol.intor('int2e')
    return mol, eri_analytical[0, 0, 0, 0]


def evaluate_density(mol, points):
    ao = mol.eval_gto("GTOval", points)
    return ao[:, 0] * ao[:, 0]


# ==========================================
# 2. Grid Generators
# ==========================================
def generate_cartesian_grid(N, L=12.0):
    """Uniform Cartesian Grid (Real or K-Space)"""
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    weights = np.full(points.shape[0], dx ** 3)
    return points, weights, dx


def generate_becke_grid(mol_coords, mol_atoms, n_radial):
    """Non-Uniform Becke Grid (Real Space)"""
    oned_grid = GaussLegendre(npoints=n_radial)
    rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned_grid)
    mgrid = MolGrid.from_preset(atnums=mol_atoms, atcoords=mol_coords, rgrid=rgrid, preset="coarse",
                                aim_weights=BeckeWeights(), store=False)

    # Filter for memory efficiency
    mask = np.linalg.norm(mgrid.points, axis=1) < 12.0
    return mgrid.points[mask], mgrid.weights[mask]


def generate_spherical_k_grid(n_rad, ang_degree):
    """Non-Uniform Spherical Grid (K-Space)"""
    x_r, w_r_gl = scipy.special.roots_legendre(n_rad)
    k_max = 30.0
    k_rad = k_max * (x_r + 1.0) / 2.0
    w_krad = w_r_gl * k_max / 2.0

    ang_grid = AngularGrid(degree=ang_degree)
    k_points = np.einsum('i,jk->ijk', k_rad, ang_grid.points).reshape(-1, 3)
    k_weights_no_k2 = np.outer(w_krad, ang_grid.weights).ravel()

    return k_points, k_weights_no_k2


# ==========================================
# 3. Integrators
# ==========================================
def solve_direct(points, weights, density):
    """Real-space generic integrator"""
    eps = 1e-7
    dist_matrix = cdist(points, points) + eps
    integrand = np.outer(density * weights, density * weights) / dist_matrix
    return np.sum(integrand), len(points)


def solve_uniform_fft(mol, N_grid, L=12.0):
    points, weights, dx = generate_cartesian_grid(N_grid, L)
    density_3d = evaluate_density(mol, points).reshape((N_grid, N_grid, N_grid))

    F_k = scipy.fft.fftshift(scipy.fft.fftn(density_3d) * (dx ** 3))
    kx = scipy.fft.fftshift(scipy.fft.fftfreq(N_grid, d=dx)) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[K2 == 0] = np.inf

    dk = kx[1] - kx[0]
    integral = np.sum((4 * np.pi / K2) * np.abs(F_k) ** 2) * (dk ** 3) / ((2 * np.pi) ** 3)
    return integral, len(points) * 2


def solve_nufft_1step(r_p, r_w, dens, k_pts, k_weights, is_spherical=False):
    x, y, z = np.ascontiguousarray(r_p[:, 0]), np.ascontiguousarray(r_p[:, 1]), np.ascontiguousarray(r_p[:, 2])
    kx, ky, kz = np.ascontiguousarray(k_pts[:, 0]), np.ascontiguousarray(k_pts[:, 1]), np.ascontiguousarray(k_pts[:, 2])
    c = np.ascontiguousarray(dens * r_w, dtype=np.complex128)

    F_k = finufft.nufft3d3(x, y, z, c, kx, ky, kz, isign=-1, eps=1e-5)

    if is_spherical:
        integral = np.sum(k_weights * 4 * np.pi * np.abs(F_k) ** 2) / (2 * np.pi ** 3)
    else:
        K2 = kx ** 2 + ky ** 2 + kz ** 2
        K2[K2 == 0] = np.inf
        integral = np.sum(k_weights * (4 * np.pi / K2) * np.abs(F_k) ** 2) / ((2 * np.pi) ** 3)

    return integral, len(r_p) + len(k_pts)


def solve_poisson_nufft_2step(r_p, r_w, dens_g, dens_f, k_pts, k_weights, is_spherical=False):
    x, y, z = np.ascontiguousarray(r_p[:, 0]), np.ascontiguousarray(r_p[:, 1]), np.ascontiguousarray(r_p[:, 2])
    kx, ky, kz = np.ascontiguousarray(k_pts[:, 0]), np.ascontiguousarray(k_pts[:, 1]), np.ascontiguousarray(k_pts[:, 2])

    c_forward = np.ascontiguousarray(dens_g * r_w, dtype=np.complex128)
    G_k = finufft.nufft3d3(x, y, z, c_forward, kx, ky, kz, isign=-1, eps=1e-5)

    if is_spherical:
        # k^2 implicitly cancelled
        c_inverse = np.ascontiguousarray(4 * np.pi * G_k * k_weights, dtype=np.complex128)
    else:
        # Must explicitly divide by K^2 for Cartesian
        K2 = kx ** 2 + ky ** 2 + kz ** 2
        K2[K2 == 0] = np.inf
        V_k = (4 * np.pi / K2) * G_k
        c_inverse = np.ascontiguousarray(V_k * k_weights, dtype=np.complex128)

    V_r_complex = finufft.nufft3d3(kx, ky, kz, c_inverse, x, y, z, isign=1, eps=1e-5)
    V_r = np.real(V_r_complex) / ((2 * np.pi) ** 3)

    eri = np.sum(dens_f * V_r * r_w)
    return eri, len(r_p) + len(k_pts)


# ==========================================
# 4. Master Benchmark Engine
# ==========================================
def run_benchmark():
    mol, exact_val = get_analytical_baseline()
    coords, atoms = mol.atom_coords(), [mol.atom_charge(i) for i in range(mol.natm)]
    data = []

    grid_params = [8, 12, 16, 20]

    for n in grid_params:
        print(f"Testing parameter scale N={n}...")

        # --- Precompute Grids for this step ---
        p_beck, w_beck = generate_becke_grid(coords, atoms, n)
        d_beck = evaluate_density(mol, p_beck)

        p_kcart, w_kcart, _ = generate_cartesian_grid(n * 2, L=30.0)
        p_ksph, w_ksph = generate_spherical_k_grid(n, n)

        def run_method(name, func, *args):
            t0 = time.perf_counter();
            tracemalloc.start()
            val, pts = func(*args)
            mem = tracemalloc.get_traced_memory()[1] / 1024 ** 2;
            tracemalloc.stop();
            t1 = time.perf_counter()
            data.append(
                {"Method": name, "Points": pts, "Error": abs(val - exact_val), "Time (s)": t1 - t0, "Memory (MB)": mem})
            gc.collect()

        # 1 & 2. Direct Methods (Capped at n=12 to prevent O(N^2) crashing)
        if n <= 12:
            p_cart, w_cart, _ = generate_cartesian_grid(n)
            d_cart = evaluate_density(mol, p_cart)
            run_method("1. Direct (Cartesian R)", solve_direct, p_cart, w_cart, d_cart)
            run_method("2. Direct (Becke R)", solve_direct, p_beck, w_beck, d_beck)

        # 3. Uniform FFT
        run_method("3. FFT (Uniform Cartesian)", solve_uniform_fft, mol, n * 2)

        # 4 & 5. 1-Step NUFFT
        run_method("4. NUFFT 1-Step (Cartesian K)", solve_nufft_1step, p_beck, w_beck, d_beck, p_kcart, w_kcart, False)
        run_method("5. NUFFT 1-Step (Spherical K)", solve_nufft_1step, p_beck, w_beck, d_beck, p_ksph, w_ksph, True)

        # 6 & 7. 2-Step Poisson NUFFT
        run_method("6. Poisson NUFFT 2-Step (Cartesian K)", solve_poisson_nufft_2step, p_beck, w_beck, d_beck, d_beck,
                   p_kcart, w_kcart, False)
        run_method("7. Poisson NUFFT 2-Step (Spherical K)", solve_poisson_nufft_2step, p_beck, w_beck, d_beck, d_beck,
                   p_ksph, w_ksph, True)

    return pd.DataFrame(data)


def plot_results(df):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Distinct colors and markers for 7 methods
    palette = {
        "1. Direct (Cartesian R)": "#d62728",
        "2. Direct (Becke R)": "#ff7f0e",
        "3. FFT (Uniform Cartesian)": "#1f77b4",
        "4. NUFFT 1-Step (Cartesian K)": "#9467bd",
        "5. NUFFT 1-Step (Spherical K)": "#2ca02c",
        "6. Poisson NUFFT 2-Step (Cartesian K)": "#e377c2",
        "7. Poisson NUFFT 2-Step (Spherical K)": "#17becf"
    }
    markers = ["o", "v", "s", "p", "*", "X", "D"]

    # Accuracy
    sns.lineplot(data=df, x="Points", y="Error", hue="Method", style="Method", markers=markers, dashes=False,
                 linewidth=2.5, ax=axes[0], palette=palette)
    axes[0].set_xscale('log')a
    axes[0].set_yscale('log')
    axes[0].set_title("Accuracy Convergence (Log-Log)")
    axes[0].set_ylabel("Absolute Error")

    # Time
    sns.lineplot(data=df, x="Points", y="Time (s)", hue="Method", style="Method", markers=markers, dashes=False,
                 linewidth=2.5, ax=axes[1], palette=palette)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title("Time Complexity (Log-Log)")

    # Memory
    sns.lineplot(data=df, x="Points", y="Memory (MB)", hue="Method", style="Method", markers=markers, dashes=False,
                 linewidth=2.5, ax=axes[2], palette=palette)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title("Space Complexity (Log-Log)")

    # Simplify legends
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df_results = run_benchmark()
    plot_results(df_results)