import argparse
import pickle

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import numpy as np


def plot_xy_from_trajectory_data_pkl(
    data,
    filename: str = "xy_distribution.png",
    draw_arrows: bool = False,
    bin_size: int = 25,
    min_count_for_arrow: int = 5,
    n_arrows: int = 40,
    arrow_scale: float = 1.0,
    min_arrow_magnitude: float = 0.05,
) -> None:
    """
    Heatmap of (x1,y1) bins. Optional clustered average arrows using (x1,y1)->(x2,y2).
    """
    xy_counts: dict[tuple[int, int], int] = {}
    arrow_sums: dict[tuple[int, int], tuple[float, float, int]] = {}

    for sample_tuple in data:
        # Expecting tuples like (s1, s2, _, _, _). Use s1[0] -> s1[1] for direction.
        s1, _s2, *_ = sample_tuple
        if len(s1) < 2:
            continue

        x1 = int(s1[0].position[0])
        y1 = int(s1[0].position[1])
        x2 = int(s1[1].position[0])
        y2 = int(s1[1].position[1])

        bx1 = x1 // bin_size
        by1 = y1 // bin_size
        bx2 = x2 // bin_size
        by2 = y2 // bin_size

        xy_counts[(bx1, by1)] = 1 + xy_counts.get((bx1, by1), 0)

        if draw_arrows:
            dxb = bx2 - bx1
            dyb = by2 - by1
            sdx, sdy, c = arrow_sums.get((bx1, by1), (0.0, 0.0, 0))
            arrow_sums[(bx1, by1)] = (sdx + dxb, sdy + dyb, c + 1)

    avg_vectors = None
    clustered_arrows = None
    if draw_arrows:
        avg_vectors = _average_arrow_vectors(arrow_sums, min_count_for_arrow)
        clustered_arrows = _cluster_average_arrows(
            avg_vectors, xy_counts, n_clusters=n_arrows, min_mag=min_arrow_magnitude
        )

    plot_xy_counts(
        xy_counts,
        filename,
        clustered_arrows=clustered_arrows,
        arrow_scale=arrow_scale,
    )


def plot_xy_from_segments(
    segments,
    filename: str = "xy_distribution.png",
    draw_arrows: bool = False,
    bin_size: int = 25,
    min_count_for_arrow: int = 5,
    n_arrows: int = 40,
    arrow_scale: float = 1.0,
    min_arrow_magnitude: float = 0.05,
) -> None:
    """
    Heatmap of segment[0] bins. Optional clustered average arrows using segment[0]->segment[1].
    """
    xy_counts: dict[tuple[int, int], int] = {}
    arrow_sums: dict[tuple[int, int], tuple[float, float, int]] = {}

    for segment in segments:
        if len(segment) < 2:
            continue

        x1 = int(segment[0].position[0])
        y1 = int(segment[0].position[1])
        x2 = int(segment[1].position[0])
        y2 = int(segment[1].position[1])

        bx1 = x1 // bin_size
        by1 = y1 // bin_size
        bx2 = x2 // bin_size
        by2 = y2 // bin_size

        xy_counts[(bx1, by1)] = 1 + xy_counts.get((bx1, by1), 0)

        if draw_arrows:
            dxb = bx2 - bx1
            dyb = by2 - by1
            sdx, sdy, c = arrow_sums.get((bx1, by1), (0.0, 0.0, 0))
            arrow_sums[(bx1, by1)] = (sdx + dxb, sdy + dyb, c + 1)

    avg_vectors = None
    clustered_arrows = None
    if draw_arrows:
        avg_vectors = _average_arrow_vectors(arrow_sums, min_count_for_arrow)
        clustered_arrows = _cluster_average_arrows(
            avg_vectors, xy_counts, n_clusters=n_arrows, min_mag=min_arrow_magnitude
        )

    plot_xy_counts(
        xy_counts,
        filename,
        clustered_arrows=clustered_arrows,
        arrow_scale=arrow_scale,
    )


def _average_arrow_vectors(
    arrow_sums: dict[tuple[int, int], tuple[float, float, int]],
    min_count_for_arrow: int,
) -> dict[tuple[int, int], tuple[float, float]]:
    """
    Turn (sum_dx, sum_dy, count) into per-bin average vectors, filtering by min_count.
    """
    averages: dict[tuple[int, int], tuple[float, float]] = {}
    for bin_key, (sdx, sdy, c) in arrow_sums.items():
        if c >= min_count_for_arrow:
            averages[bin_key] = (sdx / c, sdy / c)
    return averages


def _cluster_average_arrows(
    avg_vectors: dict[tuple[int, int], tuple[float, float]],
    counts: dict[tuple[int, int], int],
    n_clusters: int,
    min_mag: float,
) -> list[tuple[float, float, float, float]]:
    """
    Cluster bin centers and produce one weighted average arrow per cluster.
    Returns list of (center_x, center_y, dx, dy) in BIN UNITS.
    """
    if not avg_vectors:
        return []

    bins = np.array([[bx, by] for (bx, by) in avg_vectors.keys()], dtype=float)
    vecs = np.array([avg_vectors[(bx, by)] for (bx, by) in avg_vectors.keys()], dtype=float)
    wts = np.array([counts.get((bx, by), 1) for (bx, by) in avg_vectors.keys()], dtype=float)

    k = min(n_clusters, len(bins))
    if k < 1:
        return []

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(bins)

    clustered: list[tuple[float, float, float, float]] = []
    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            continue
        w = wts[mask]
        pos = bins[mask]
        v = vecs[mask]
        wsum = float(w.sum())
        cx, cy = (pos * w[:, None]).sum(axis=0) / wsum
        vx, vy = (v * w[:, None]).sum(axis=0) / wsum

        mag = float(np.hypot(vx, vy))
        if mag >= min_mag:
            clustered.append((cx, cy, vx, vy))

    return clustered


def plot_xy_counts(
    xy_counts: dict[tuple[int, int], int],
    filename: str = "xy_distribution.png",
    clustered_arrows: list[tuple[float, float, float, float]] | None = None,
    arrow_scale: float = 1.0,
) -> None:
    """
    Plot scatter heatmap of bin counts. Optionally overlay clustered arrows (one per cluster).
    """
    if not xy_counts:
        print("No data to plot.")
        return

    unique_xy_pairs = list(xy_counts.keys())
    counts = list(xy_counts.values())
    x_unique, y_unique = zip(*unique_xy_pairs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_unique, y_unique, c=counts, cmap="viridis", s=100)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Occurrences")

    plt.xlabel("X (binned)")
    plt.ylabel("Y (binned)")
    plt.title("XY Distribution (counts per bin)")

    if clustered_arrows:
        qx, qy, qu, qv = [], [], [], []
        for cx, cy, vx, vy in clustered_arrows:
            qx.append(cx)
            qy.append(cy)
            qu.append(vx * arrow_scale)
            qv.append(vy * arrow_scale)

        plt.quiver(
            qx, qy, qu, qv,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
            alpha=0.95,
        )

    plt.savefig(filename, dpi=600)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XY distribution with optional clustered arrows.")
    parser.add_argument(
        "--pkl",
        type=str,
        default="database_1000000_pairs_1_rules_1_length.pkl",
        help="Path to PKL file with trajectory tuples.",
    )
    parser.add_argument("--out", type=str, default="xy_distribution.png", help="Output image filename.")
    parser.add_argument("--bin-size", type=int, default=25, help="Bin size in pixels.")
    parser.add_argument("--arrrow", action="store_true", help="Overlay clustered average arrows.")
    parser.add_argument("--n-arrows", type=int, default=40, help="Number of arrow clusters.")
    parser.add_argument("--min-arrow-count", type=int, default=5, help="Min samples in a bin to consider for arrows.")
    parser.add_argument("--min-arrow-mag", type=float, default=0.05, help="Min avg vector magnitude (in bin units).")
    parser.add_argument("--arrow-scale", type=float, default=1.0, help="Scale factor applied to arrow vectors.")
    args = parser.parse_args()

    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)

    plot_xy_from_trajectory_data_pkl(
        data,
        filename=args.out,
        draw_arrows=args.arrrow,
        bin_size=args.bin_size,
        min_count_for_arrow=args.min_arrow_count,
        n_arrows=args.n_arrows,
        arrow_scale=args.arrow_scale,
        min_arrow_magnitude=args.min_arrow_mag,
    )


if __name__ == "__main__":
    main()
