"""Build a synthetic GemPy model and export it to CMB.

The model has two parallel contacts that dip toward positive x (z decreases
with x). The GemPy result is sampled at the cell centers of a discretize
TensorMesh and stored as a categorical lithology model plus a
lithology-derived resistivity model. Run this file from the repository root
after installing the optional dependencies described in
``examples/README.md``.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

import cmb_format as cmb

EXTENT = np.array([0.0, 100.0, 0.0, 60.0, -60.0, 40.0])
RESOLUTION = np.array([32, 16, 24])
SLOPE = -0.35
UPPER_INTERCEPT = 4.0
LOWER_INTERCEPT = -20.0

LITHOLOGY = {
    1: {"name": "young_cover", "resistivity_ohm_m": 100.0},
    2: {"name": "middle_unit", "resistivity_ohm_m": 50.0},
    3: {"name": "basement", "resistivity_ohm_m": 10.0},
}


def _contact_z(x: np.ndarray, intercept: float) -> np.ndarray:
    """Return the z coordinate of a synthetic planar contact."""
    return SLOPE * x + intercept


def _build_gempy_model():
    """Create the self-contained two-contact GemPy model."""
    import gempy as gp

    x = np.array([0.0, 50.0, 100.0])
    y = np.array([0.0, 30.0, 60.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    x_points = np.concatenate([xx.ravel(), xx.ravel()])
    y_points = np.concatenate([yy.ravel(), yy.ravel()])
    z_points = np.concatenate(
        [
            _contact_z(xx.ravel(), UPPER_INTERCEPT),
            _contact_z(xx.ravel(), LOWER_INTERCEPT),
        ]
    )
    surface_names = ["upper_contact"] * 9 + ["lower_contact"] * 9
    name_id_map = {"upper_contact": 1, "lower_contact": 2}

    surface_points = gp.data.SurfacePointsTable.from_arrays(
        x=x_points,
        y=y_points,
        z=z_points,
        names=surface_names,
        name_id_map=name_id_map,
    )
    orientations = gp.data.OrientationsTable.from_arrays(
        x=np.array([50.0, 50.0]),
        y=np.array([30.0, 30.0]),
        z=np.array(
            [
                _contact_z(np.array([50.0]), UPPER_INTERCEPT)[0],
                _contact_z(np.array([50.0]), LOWER_INTERCEPT)[0],
            ]
        ),
        G_x=np.array([-SLOPE, -SLOPE]),
        G_y=np.array([0.0, 0.0]),
        G_z=np.array([1.0, 1.0]),
        names=["upper_contact", "lower_contact"],
        name_id_map=name_id_map,
    )
    frame = gp.data.StructuralFrame.from_data_tables(surface_points, orientations)
    return gp.create_geomodel(
        project_name="synthetic_dipping_layers",
        extent=EXTENT,
        resolution=RESOLUTION,
        structural_frame=frame,
    )


def _evaluate_lithology(model, points: np.ndarray) -> np.ndarray:
    """Evaluate GemPy and normalize the current custom-grid return shape."""
    import gempy as gp
    from gempy.core.data import GemPyEngineConfig
    from gempy_engine.config import AvailableBackends

    # Keep the demo portable across machines with or without a GPU.
    engine_config = GemPyEngineConfig(
        backend=AvailableBackends.numpy,
        use_gpu=False,
    )
    result = gp.compute_model_at(model, at=points, engine_config=engine_config)
    # GemPy 2026 returns the custom array directly. Keep this fallback for
    # versions whose public API returns a Solutions object instead.
    if hasattr(result, "raw_arrays"):
        result = result.raw_arrays.custom
    result = np.asarray(result)
    if result.ndim != 1 or result.shape[0] != points.shape[0]:
        raise ValueError(
            f"GemPy custom output must be shaped ({points.shape[0]},), "
            f"got {result.shape}"
        )
    if not np.all(np.isfinite(result)) or not np.allclose(result, np.rint(result)):
        raise ValueError("GemPy returned non-integral or non-finite lithology IDs")
    return np.rint(result).astype(np.int32)


def _check_known_layers(model) -> None:
    """Check GemPy IDs at points well away from the two known contacts."""
    probes = np.array(
        [
            [20.0, 30.0, 30.0],  # above upper contact (z = -3)
            [20.0, 30.0, -12.0],  # between contacts (z = -27)
            [20.0, 30.0, -50.0],  # below lower contact
            [80.0, 30.0, 35.0],  # above upper contact (z = -24)
            [80.0, 30.0, -35.0],  # between contacts (lower z = -48)
            [80.0, 30.0, -58.0],  # below lower contact
        ]
    )
    expected = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
    actual = _evaluate_lithology(model, probes)
    np.testing.assert_array_equal(actual, expected)


def _write_plot(mesh, lithology: np.ndarray, output_path: Path) -> None:
    """Write an x-z lithology section through the middle y index."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    nx, ny, nz = (len(axis) for axis in mesh.h)
    values = lithology.reshape((nx, ny, nz), order="F")
    y_index = ny // 2
    section = values[:, y_index, :].T
    x_edges = np.r_[mesh.origin[0], mesh.origin[0] + np.cumsum(mesh.h[0])]
    z_edges = np.r_[mesh.origin[2], mesh.origin[2] + np.cumsum(mesh.h[2])]

    colors = ["#4c78a8", "#f2cf5b", "#7f5a3b"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.pcolormesh(x_edges, z_edges, section, cmap=cmap, norm=norm, shading="flat")
    ax.set(
        xlabel="x (m)",
        ylabel="z (m)",
        title="GemPy synthetic dipping layers (middle y section)",
    )
    ax.set_xlim(mesh.origin[0], x_edges[-1])
    ax.set_ylim(z_edges[0], z_edges[-1])
    x_line = np.linspace(x_edges[0], x_edges[-1], 200)
    ax.plot(
        x_line,
        _contact_z(x_line, UPPER_INTERCEPT),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    ax.plot(
        x_line,
        _contact_z(x_line, LOWER_INTERCEPT),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    legend_handles = [
        Patch(color=colors[code - 1], label=f"{info['name']} (ID {code})")
        for code, info in LITHOLOGY.items()
    ]
    legend_handles.append(
        Line2D([], [], color="black", linestyle="--", label="input contacts")
    )
    ax.legend(handles=legend_handles, loc="lower left", fontsize="small")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(output_dir: Path, make_plot: bool = True) -> tuple[Path, Path | None]:
    """Build, export, read, and verify the example model."""
    import discretize
    import gempy

    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_gempy_model()
    _check_known_layers(model)

    h = [
        np.full(int(n), (hi - lo) / n)
        for n, (lo, hi) in zip(RESOLUTION, EXTENT.reshape(3, 2), strict=True)
    ]
    origin = EXTENT[[0, 2, 4]]
    mesh = discretize.TensorMesh(h=h, origin=origin)
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    lithology = _evaluate_lithology(model, centers)
    resistivity = np.array(
        [LITHOLOGY[int(code)]["resistivity_ohm_m"] for code in lithology],
        dtype=np.float64,
    )

    mapping = {
        str(code): {
            "name": info["name"],
            "resistivity_ohm_m": info["resistivity_ohm_m"],
        }
        for code, info in LITHOLOGY.items()
    }
    try:
        cmb_package_version = version("cmb-format")
    except PackageNotFoundError:
        cmb_package_version = "source-checkout"

    mesh_data = {
        "mode": "embedded",
        "mesh_class": "TensorMesh",
        "arrays": {
            "h_x": np.asarray(mesh.h[0]),
            "h_y": np.asarray(mesh.h[1]),
            "h_z": np.asarray(mesh.h[2]),
            "origin": np.asarray(mesh.origin),
        },
    }
    models = {
        "lithology_id": {
            "metadata": {
                "kind": "categorical",
                "units": "dimensionless",
                "description": "GemPy lithology enumeration at cell centers",
                "lithology_mapping": mapping,
            },
            "array": lithology,
        },
        "resistivity": {
            "metadata": {
                "kind": "physical_property",
                "units": "ohm-m",
                "description": "Constant synthetic resistivity by lithology",
                "lithology_mapping": mapping,
            },
            "array": resistivity,
        },
    }
    metadata = {
        "description": "Synthetic two-contact dipping-layer model",
        "provenance": {
            "generator": "examples/gempy_to_cmb.py",
            "gempy_version": gempy.__version__,
            "discretize_version": discretize.__version__,
            "cmb_package_version": cmb_package_version,
            "cmb_file_format_version": cmb.WRITTEN_FORMAT_VERSION,
        },
        "coordinate_system": {"axes": ["x", "y", "z"], "units": "m"},
        "cell_center_sampling": True,
    }

    cmb_path = output_dir / "gempy_dipping_layers.cmb"
    cmb.write_file(cmb_path, mesh_data, models, metadata)
    read_mesh, read_models, read_metadata = cmb.read_file(cmb_path)
    for axis in ("h_x", "h_y", "h_z", "origin"):
        np.testing.assert_array_equal(
            read_mesh["arrays"][axis], mesh_data["arrays"][axis]
        )
    for name in models:
        np.testing.assert_array_equal(read_models[name]["array"], models[name]["array"])
        assert read_models[name]["metadata"] == models[name]["metadata"]
    assert read_metadata == metadata

    roundtrip_mesh = discretize.TensorMesh(
        h=[read_mesh["arrays"][name] for name in ("h_x", "h_y", "h_z")],
        origin=read_mesh["arrays"]["origin"],
    )
    assert mesh.equals(roundtrip_mesh)

    plot_path = None
    if make_plot:
        plot_path = output_dir / "gempy_dipping_layers.png"
        _write_plot(
            roundtrip_mesh,
            np.asarray(read_models["lithology_id"]["array"]),
            plot_path,
        )
    print(f"Wrote and verified {cmb_path} ({mesh.nC} cells)")
    if plot_path is not None:
        print(f"Wrote {plot_path}")
    return cmb_path, plot_path


def main() -> None:
    """Parse command-line arguments and run the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output"),
        help="directory for generated CMB and PNG files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the optional Matplotlib cross-section",
    )
    args = parser.parse_args()
    run(args.output_dir, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
