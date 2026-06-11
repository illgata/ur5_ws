#!/usr/bin/env python3
"""Prepare the labeled Gazebo box dataset and train a YOLO detector."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ALIASES = {
    "red": "red_box",
}


def read_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_image_label_pairs(dataset_root: Path) -> list[tuple[str, Path, Path, list[str]]]:
    pairs: list[tuple[str, Path, Path, list[str]]] = []
    for image_group in sorted(dataset_root.iterdir()):
        if not image_group.is_dir() or image_group.name.endswith("_label"):
            continue
        label_group = dataset_root / f"{image_group.name}_label"
        for split in ("train", "val"):
            image_dir = image_group / split
            label_dir = label_group / split
            if not image_dir.is_dir():
                continue
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Missing label directory for {image_dir}: {label_dir}")
            local_classes = read_classes(label_dir / "classes.txt")
            if not local_classes:
                local_classes = read_classes(label_group / "classes.txt")
            if not local_classes:
                local_classes = [image_group.name]
            pairs.append((split, image_dir, label_dir, local_classes))
    return pairs


def remap_label_file(
    src: Path,
    dst: Path,
    local_classes: list[str],
    global_index: dict[str, int],
    aliases: dict[str, str],
) -> tuple[int, int]:
    object_count = 0
    bad_count = 0
    output_lines: list[str] = []

    for line_no, raw_line in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            bad_count += 1
            print(f"[WARN] Bad label line ignored: {src}:{line_no}: {raw_line}")
            continue
        try:
            local_id = int(float(parts[0]))
        except ValueError:
            bad_count += 1
            print(f"[WARN] Bad class id ignored: {src}:{line_no}: {raw_line}")
            continue
        if local_id < 0 or local_id >= len(local_classes):
            bad_count += 1
            print(f"[WARN] Class id {local_id} outside {local_classes}: {src}:{line_no}")
            continue

        class_name = aliases.get(local_classes[local_id], local_classes[local_id])
        if class_name not in global_index:
            bad_count += 1
            print(f"[WARN] Class {class_name!r} is not in root classes.txt: {src}:{line_no}")
            continue
        output_lines.append(" ".join([str(global_index[class_name]), *parts[1:]]))
        object_count += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return object_count, bad_count


def prepare_dataset(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    classes = read_classes(dataset_root / "classes.txt")
    if not classes:
        raise FileNotFoundError(f"Missing or empty class list: {dataset_root / 'classes.txt'}")

    aliases = dict(DEFAULT_ALIASES)
    for item in args.alias:
        if "=" not in item:
            raise ValueError(f"Alias must be old=new, got: {item}")
        old, new = item.split("=", 1)
        aliases[old.strip()] = new.strip()

    global_index = {name: idx for idx, name in enumerate(classes)}
    pairs = find_image_label_pairs(dataset_root)
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {dataset_root}")

    if output_root.exists() and args.rebuild:
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0
    missing_labels = 0
    bad_labels = 0

    for split, image_dir, label_dir, local_classes in pairs:
        group_name = image_dir.parent.name
        print(f"[DATA] {group_name}/{split}: local classes={local_classes}")
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            out_name = f"{group_name}_{image_path.name}"
            label_path = label_dir / f"{image_path.stem}.txt"
            out_image = output_root / "images" / split / out_name
            out_label = output_root / "labels" / split / f"{Path(out_name).stem}.txt"
            shutil.copy2(image_path, out_image)
            if label_path.exists():
                objects, bad = remap_label_file(label_path, out_label, local_classes, global_index, aliases)
                object_count += objects
                bad_labels += bad
            else:
                missing_labels += 1
                out_label.write_text("", encoding="utf-8")
                print(f"[WARN] Missing label for image, wrote empty label: {image_path}")
            image_count += 1

    data_yaml = output_root / "data.yaml"
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                f"nc: {len(classes)}",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Prepared YOLO dataset: {output_root}")
    print(f"[OK] data.yaml: {data_yaml}")
    print(f"[OK] images={image_count}, objects={object_count}, missing_labels={missing_labels}, bad_labels={bad_labels}")
    print(f"[OK] global classes={classes}")
    return data_yaml


def train(args: argparse.Namespace, data_yaml: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found at {best}")
    print(f"[OK] Best model: {best}")

    if args.deploy_model:
        deploy_path = args.deploy_model.resolve()
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, deploy_path)
        print(f"[OK] Copied best.pt to project detector model: {deploy_path}")

    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("train_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("yolo_training/grasp_objects"))
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model or a .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Use '0' for GPU 0 or 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="grasp_objects")
    parser.add_argument("--alias", action="append", default=[], help="Class name alias, e.g. red=red_box")
    parser.add_argument("--prepare-only", action="store_true", help="Only build YOLO dataset and data.yaml")
    parser.add_argument("--rebuild", action="store_true", default=True, help="Rebuild prepared dataset directory")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing training run directory")
    parser.add_argument(
        "--deploy-model",
        type=Path,
        default=Path("src/vision/vision/yolov11/models/best.pt"),
        help="Copy trained best.pt here after training. Use '' to disable.",
    )
    args = parser.parse_args()
    if args.deploy_model is not None and str(args.deploy_model) == "":
        args.deploy_model = None
    return args


def main() -> int:
    args = parse_args()
    data_yaml = prepare_dataset(args)
    if args.prepare_only:
        return 0
    train(args, data_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
