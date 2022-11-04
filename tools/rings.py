import numpy as np
from matplotlib import pyplot as plt

from crings import fast_lidar_to_rings


def generate_rings(
    angle_levels=64,
    range_levels=64,
    expansion_term=3.12,
    min_resolution=0.01,
    min_dist=0.3,
    max_dist=25.0,
    VISUALIZE=False,
):
    """
    expansion_term: 0. (equal range levels) 1. (linear increase in depths), >1. (exponential)

    range levels
    ---
    first bin is           0          to   min_dist
    second bin is          min_dist   to   min_resolution
    nth bin is             x          to   x
    ...
    second-to-last bin is  x          to   max_dist
    last bin is            max_dist   to   inf

    bin values
    ---
    0 points in bin     0
    1 points in bin     0.5
    2 points in bin     0.75
    ..
    inf points in bin   1.
    (a.k.a x(n) = 1-2^-n)
    """
    x = np.linspace(0, 1, range_levels - 2)
    expansion_curve = np.power(x, expansion_term)  # a.k.a range_level_depths
    renormalisation_factor = np.sum(expansion_curve) / (
        max_dist - (min_resolution * (range_levels - 2)) - min_dist
    )
    range_level_depths = expansion_curve / renormalisation_factor + min_resolution
    range_level_maxs = np.cumsum(range_level_depths) + min_dist
    range_level_maxs = np.concatenate([[min_dist], range_level_maxs, [np.inf]]).astype(np.float32)
    range_level_mins = np.concatenate([[0.0], range_level_maxs[:-1]]).astype(np.float32)

    if VISUALIZE:
        th = np.linspace(-7.0 / 8.0 * np.pi, 7.0 / 8.0 * np.pi, angle_levels)
        plt.figure("curve")
        plt.plot(range_level_maxs[:-1])
        for x, y in enumerate(range_level_maxs[:-1]):
            plt.axhline(y)
            plt.axvline(x)
        plt.figure("rings")
        for i, r in enumerate(range_level_maxs[:-1]):
            plt.gca().add_artist(plt.Circle((0, 0), r, color="k", fill=False))
        for i in range(angle_levels):
            plt.plot(
                [min_dist * np.cos(th), r * np.cos(th)],
                [min_dist * np.sin(th), r * np.sin(th)],
                "k",
            )
            plt.axis("equal")
        plt.gca().add_artist(plt.Circle((1, 1), 0.3, color="r", zorder=3))
        plt.tight_layout()

    def lidar_to_rings(scans):
        """
        scans: ndarray (n, N_RAYS)   0-100 [m]
        rings: ndarray (n_scans, angle_levels, range_levels, n_channels)
        """
        return fast_lidar_to_rings(scans, angle_levels, range_levels, range_level_mins, range_level_maxs)

    def old_lidar_to_rings(scans):
        """
        scans: ndarray (n, N_RAYS)   0-100 [m]
        rings: ndarray (n_scans, angle_levels, range_levels, n_channels)
        """
        # remove 0 returns
        scans = scans * 1.0
        scans[scans == 0] = np.inf

        CHANNEL = 0
        N_RAYS = scans.shape[1]
        i_to_j, j_to_ii = generate_downsampling_map(N_RAYS, angle_levels)

        rings = np.zeros(
            (scans.shape[0], angle_levels, range_levels, 1), dtype=np.uint8
        )
        # count level hits for each angular section
        # j is index of angular section, ii are indices of corresponding rays
        for j, ii in enumerate(j_to_ii):
            is_unseen = scans[:, ii, None] < range_level_mins[None, None, :]
            is_hit = np.logical_and(
                scans[:, ii, None] < range_level_maxs[None, None, :],
                scans[:, ii, None] >= range_level_mins[None, None, :],
            )
            #             rings[:,j,:,CHANNEL] = np.sum(is_hit, axis=1)
            #             rings[:,j,:,CHANNEL] = 1 * np.all(is_unseen, axis=1) + 2 * np.any(is_hit, axis=1)
            rings[:, j, :, CHANNEL] = np.clip(
                1 * np.any(is_unseen, axis=1) + 2 * np.any(is_hit, axis=1), 0, 2
            )
        return rings

    def rings_to_lidar(rings, N_RAYS=1080):
        CHANNEL = 0
        i_to_j, j_to_ii = generate_downsampling_map(N_RAYS, angle_levels)
        scans = np.zeros((rings.shape[0], N_RAYS), dtype=np.float32)
        for j, ii in enumerate(j_to_ii):
            scans[:, ii] = range_level_mins[np.argmax(rings[:, j, :, :] > 0.4, axis=1)][
                :, None, CHANNEL
            ]
        return scans

    def visualize_rings(
        ring, scan=None, angle_min=0, angle_max=2 * np.pi, fig=None, ax=None, plot_regen=False
    ):
        CHANNEL = 0
        th = np.linspace(angle_min, angle_max, angle_levels)
        r = range_level_mins
        thth, rr = np.meshgrid(th, r)
        if fig is None:
            fig = plt.figure("rings")
        if ax is None:
            ax = fig.add_subplot(111, projection="polar")
        ax.clear()
        ax.pcolormesh(thth, rr, ring[:, :, CHANNEL].T, cmap=plt.cm.Greys)
        if scan is not None:
            scan_regen = rings_to_lidar(ring[None, :, :, :], scan.shape[0])[0, :]
            scan_th = np.linspace(angle_min, angle_max, scan.shape[0])
            plt.plot(scan_th, scan, "r")
            if plot_regen:
                plt.plot(scan_th, scan_regen, "g")
        return ax

    return {
        "range_level_mins": range_level_mins,
        "range_level_maxs": range_level_maxs,
        "lidar_to_rings": lidar_to_rings,
        "rings_to_lidar": rings_to_lidar,
        "visualize_rings": visualize_rings,
        "rings_to_bool": 2.0,
    }


def generate_downsampling_map(I, J):
    """
    with,
    I = 5
    J = 2

    0   1               I
    |___|___|___|___|___|
    |_________|_________|
    0         1         J

    >> i_to_j, j_to_ii = downsample_map(5, 2)
    >> i_to_j
    [0, 0, 0, 1, 1]
    >> j_to_ii[0]
    [0, 1, 2]
    >> j_to_ii[1]
    [3, 4]
    """
    downsample_factor = I * 1.0 / J
    i_to_j = np.floor(np.arange(I) / downsample_factor).astype(int)
    j_to_ii = [
        np.arange(
            np.ceil(j * downsample_factor),
            np.ceil((j + 1) * downsample_factor),
            dtype=int,
        )
        for j in range(J)
    ]
    return i_to_j, j_to_ii


if __name__ == "__main__":
    ring_def = generate_rings()
    # linear ramp
    scans = np.ones((1, 1080)) * (np.arange(1080) / 1080.0 * 25.0)[None, :]
    rings = ring_def["lidar_to_rings"](scans.astype(np.float32))
    plt.ion()
    ring_def["visualize_rings"](rings[0, :, :, :], scan=scans[0])
