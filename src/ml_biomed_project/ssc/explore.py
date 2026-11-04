"""Small, reporting-only tools for the SSC notebook; no inference on widget changes."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ml_biomed_project.metrics import dice_score
from ml_biomed_project.ssc.inference import CHECKPOINT_SHA256

COLORS = {"Reference": "#dd6596", "SSC4": "#168e99"}


@dataclass
class Demo:
    """A relocatable local folder: manifest, saved scores, example scans and model."""

    root: Path
    manifest: dict

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Demo paths must stay inside the bundle.")
        return path

    @property
    def cases(self):
        return pd.DataFrame(
            [
                {k: row[k] for k in ("case_id", "sequence", "split", "selection")}
                for row in self.manifest["cases"]
            ]
        )

    @property
    def scores(self):
        return pd.read_csv(self.path("scores.csv"))

    @property
    def model_dir(self):
        return self.path("model")

    def row(self, case_id):
        return next(row for row in self.manifest["cases"] if row["case_id"] == case_id)

    def load_case(self, case_id):
        row = self.row(case_id)
        return load_case(
            self.path(row["image"]),
            self.path(row["reference"]),
            {
                name: (self.path(item["path"]), item["label"])
                for name, item in row["predictions"].items()
            },
            name=case_id,
        )


def load_demo(folder):
    folder = Path(folder).resolve()
    if not (folder / "manifest.json").is_file():
        raise FileNotFoundError(
            f"No SSC demo bundle at {folder}. See section 1 of notebooks/ssc-domain-generalization.ipynb "
            "or place the shared bundle here. Run ensure_demo(folder) to fetch missing files."
        )
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise ValueError("Expected SSC demo manifest version 1 with example cases.")
    model = manifest.get("model", {})
    if (
        model.get("method") != "SSC4"
        or model.get("input_features") != 4
        or model.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or any(
            set(row.get("predictions", {})) != {"SSC4"}
            or row["predictions"]["SSC4"].get("label") != 1
            for row in manifest["cases"]
        )
    ):
        raise ValueError(
            "Incompatible demo bundle. Regenerate the SSC4 bundle with scripts/prepare_ssc_demo.py into a fresh folder."
        )
    return Demo(folder, manifest)


def load_case(image_path, reference_path=None, predictions=None, *, name=None):
    """Read native grids, score ALL voxels, then permute/flip identically for display.

    predictions maps display names to (NIfTI path, foreground label). No resampling,
    threshold fitting or connected-component removal. A reference is optional.
    """
    image = nib.load(image_path)
    if len(image.shape) != 3:
        raise ValueError("A single 3D MRI volume is required.")
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Image contains nonfinite values.")
    masks = {}
    sources = {}
    if reference_path is not None:
        sources["Reference"] = (reference_path, 1)
    sources.update(predictions or {})
    if "Reference" in (predictions or {}):
        raise ValueError("Reference is a reserved display name.")
    for label, (path, foreground) in sources.items():
        obj = nib.load(path)
        if obj.shape != image.shape or not np.allclose(obj.affine, image.affine, atol=1e-5, rtol=0):
            raise ValueError(f"Native image/mask grid mismatch: {label}. No automatic repair.")
        data = np.asanyarray(obj.dataobj)
        if not np.isfinite(data).all() or not np.equal(data, np.rint(data)).all():
            raise ValueError(f"Expected finite discrete labels for {label}.")
        if label in {"Reference", "SSC4"} and not np.isin(data, [0, 1]).all():
            raise ValueError(f"{label} must be binary 0/1.")
        masks[label] = data == foreground
    scores = []
    if "Reference" in masks:
        truth = masks["Reference"]
        for label, mask in masks.items():
            if label == "Reference":
                continue
            tp = int(np.count_nonzero(mask & truth))
            fp = int(np.count_nonzero(mask & ~truth))
            fn = int(np.count_nonzero(~mask & truth))
            scores.append(
                {
                    "model": label,
                    "Dice": dice_score(mask, truth),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "recall": tp / (tp + fn) if tp + fn else np.nan,
                }
            )
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )
    canonical = nib.as_closest_canonical(image)
    nonzero = values[values != 0]
    window = tuple(np.percentile(nonzero, [1, 99])) if nonzero.size else (0, 1)
    return {
        "name": name or Path(image_path).name,
        "image_path": Path(image_path),
        "image": nib.orientations.apply_orientation(values, orientation),
        "masks": {k: nib.orientations.apply_orientation(v, orientation) for k, v in masks.items()},
        "spacing": nib.affines.voxel_sizes(canonical.affine),
        "window": window,
        "scores": pd.DataFrame(scores),
        "native_shape": image.shape,
        "native_axis_codes": nib.aff2axcodes(image.affine),
    }


def plot_case(case, plane="Axial", index=None, alpha=0.25):
    """Synchronized full-field slices; reported Dice remains a whole-volume score."""
    axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[plane]
    index = case["image"].shape[axis] // 2 if index is None else int(index)
    if not 0 <= index < case["image"].shape[axis]:
        raise ValueError("Slice index is outside this volume.")
    remaining = [i for i in range(3) if i != axis]
    aspect = case["spacing"][remaining[1]] / case["spacing"][remaining[0]]
    panels = ["MRI", *case["masks"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), squeeze=False)
    scores = case["scores"].set_index("model") if not case["scores"].empty else None
    truth = case["masks"].get("Reference")
    for ax, label in zip(axes[0], panels, strict=True):
        ax.imshow(
            np.take(case["image"], index, axis=axis).T,
            cmap="gray",
            origin="lower",
            vmin=case["window"][0],
            vmax=case["window"][1],
            aspect=aspect,
        )
        if label != "MRI":
            mask = np.take(case["masks"][label], index, axis=axis).T
            color = COLORS.get(label, "#5286c5")
            ax.imshow(
                np.ma.masked_where(~mask, np.ones(mask.shape)),
                origin="lower",
                cmap=ListedColormap([color]),
                alpha=alpha,
                interpolation="nearest",
                aspect=aspect,
            )
            if mask.any() and not mask.all():
                ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.2)
        if label not in ("MRI", "Reference") and truth is not None:
            ref = np.take(truth, index, axis=axis).T
            if ref.any() and not ref.all():
                ax.contour(ref, levels=[0.5], colors=[COLORS["Reference"]], linestyles="dashed")
        title = label
        if scores is not None and label in scores.index:
            title += f"\nVolume Dice {scores.loc[label, 'Dice']:.3f}"
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"{case['name']} · {plane} index {index} · canonical array coordinates")
    fig.tight_layout()
    return fig


def case_viewer(demo, initial_case=None, initial_index=None):
    """Choose cases/planes/slices without rerunning a model or caching all MRIs."""
    import ipywidgets as widgets
    from IPython.display import display

    options = list(demo.cases.case_id)
    choice = widgets.Dropdown(options=options, value=initial_case or options[0], description="Case")
    plane = widgets.Dropdown(options=["Axial", "Coronal", "Sagittal"], description="Plane")
    slider = widgets.IntSlider(description="Slice", continuous_update=False)
    opacity = widgets.FloatSlider(
        value=0.25, min=0, max=0.7, step=0.05, description="Opacity", continuous_update=False
    )
    output = widgets.Output()
    current = demo.load_case(choice.value)
    busy = False

    def reset():
        nonlocal busy
        busy = True
        axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[plane.value]
        slider.max = current["image"].shape[axis] - 1
        slider.value = slider.max // 2
        busy = False

    def render(change=None):
        if busy:
            return
        with output:
            output.clear_output(wait=True)
            fig = plot_case(current, plane.value, slider.value, opacity.value)
            display(fig)
            plt.close(fig)

    def select_case(change):
        nonlocal current
        current = demo.load_case(change["new"])
        reset()
        render()

    reset()
    if initial_index is not None:
        if not 0 <= initial_index <= slider.max:
            raise ValueError("Initial slice index is outside the chosen volume.")
        slider.value = initial_index
    choice.observe(select_case, names="value")
    plane.observe(lambda change: (reset(), render()), names="value")
    slider.observe(render, names="value")
    opacity.observe(render, names="value")
    render()
    return widgets.VBox([widgets.HBox([choice, plane]), widgets.HBox([slider, opacity]), output])


def plot_study(scores):
    """Full retained SSC cohorts, one point per operational group, not demo-only cases."""
    grouped = scores.groupby(["sequence", "patient_group"], sort=True).dice.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    rng = np.random.default_rng(1729)
    for position, sequence in enumerate(["T1", "T2"]):
        values = grouped.loc[grouped.sequence == sequence, "dice"].to_numpy()
        ax.scatter(
            position + rng.uniform(-0.13, 0.13, values.size),
            values,
            color=["#666e9d", "#168e99"][position],
            alpha=0.5,
            s=17,
        )
        ax.hlines(values.mean(), position - 0.2, position + 0.2, color="black", linewidth=2)
    ax.set(
        xticks=[0, 1],
        xticklabels=[
            f"{sequence} {'/'.join(scores.loc[scores.sequence == sequence, 'split'].unique())}"
            for sequence in ["T1", "T2"]
        ],
        ylabel="Whole-volume Dice",
        ylim=(-0.03, 1.03),
        title="SSC4 · full retained cohorts",
    )
    fig.tight_layout()
    return fig


def study_summary(scores):
    rows = []
    for sequence, scans in scores.groupby("sequence"):
        groups = scans.groupby("patient_group").dice.mean()
        rows.append(
            {
                "sequence": sequence,
                "scans": len(scans),
                "operational groups": len(groups),
                "group-mean Dice": groups.mean(),
                "scan-mean Dice": scans.dice.mean(),
                "scan median": scans.dice.median(),
                "scans Dice < 0.10": int((scans.dice < 0.1).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(history.epoch, history.train_loss, label="Train")
    axes[0].plot(history.epoch, history.val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Native compound loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.patch_dice, label="Patch Dice")
    axes[1].plot(history.epoch, history.ema_dice, label="EMA")
    axes[1].set(xlabel="Epoch", ylabel="Validation patch Dice", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    return fig
