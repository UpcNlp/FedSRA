#!/usr/bin/env python3
"""Co-Boosting (ICLR 2024) on the real-source fundus benchmark.

The script consumes the CE client checkpoints produced by
``realfed_fundus.py`` and retains the official Co-Boosting synthesizer,
ensemble-weight adaptation, hard-sample loss, ODS perturbation, and data-free
KL distillation.  Synthetic images are generated at a configurable lower
resolution (64 by default) because ResNet-18 is fully convolutional; all real
validation/test images remain at the shared 224-pixel evaluation resolution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from realfed_fundus import (
    SOURCES,
    CEModel,
    FundusDataset,
    atomic_json_dump,
    atomic_torch_save,
    binary_metrics,
    build_transform,
    git_revision,
    load_sources,
    make_loader,
    optimizer_to,
    seed_everything,
    state_to_fp16_cpu,
    subset_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COBOOST_ROOT = REPO_ROOT / "Co-Boosting-main"
sys.path.insert(0, str(COBOOST_ROOT))
import datafree  # noqa: E402
from datafree.synthesis.coboost import COBOOSTSynthesizer  # noqa: E402
from utils_fl import WEnsemble  # noqa: E402

# Importing ``datafree.models`` executes every legacy classifier module, one of
# which depends on the removed ``torchvision.models.utils`` namespace.  Load
# the official generator source directly so that unrelated legacy models do
# not prevent the current baseline from running on modern torchvision.
_generator_spec = importlib.util.spec_from_file_location(
    "coboost_official_generator", COBOOST_ROOT / "datafree/models/generator.py"
)
if _generator_spec is None or _generator_spec.loader is None:
    raise ImportError("Could not load the official Co-Boosting generator")
_generator_module = importlib.util.module_from_spec(_generator_spec)
_generator_spec.loader.exec_module(_generator_module)
Generator = _generator_module.Generator


@dataclass(frozen=True)
class Config:
    seed: int
    epochs: int
    image_size: int
    synth_size: int
    heldout: str

    @property
    def clients(self) -> List[str]:
        return [source for source in SOURCES if source != self.heldout]

    @property
    def tag(self) -> str:
        heldout = self.heldout if self.heldout else "none"
        return f"realfed_binary_coboost_heldout-{heldout}_s{self.seed}"

    @property
    def teacher_tag(self) -> str:
        heldout = self.heldout if self.heldout else "none"
        return f"realfed_binary_ce_heldout-{heldout}_s{self.seed}"


class TeacherAdapter(nn.Module):
    """Expose CEModel through the interface expected by official WEnsemble."""

    def __init__(self, model: CEModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ):
        feature = self.model.backbone.forward_raw(x)
        logits = self.model.classifier(feature)
        if return_features:
            return logits, feature
        return logits


class PrintLogger:
    def info(self, message, *args) -> None:
        if args:
            message = message % args
        print(message, flush=True)


def load_teachers(
    cfg: Config, teacher_output: Path, device: torch.device
) -> Tuple[List[TeacherAdapter], List[Dict[str, object]]]:
    teachers = []
    metas = []
    checkpoint_dir = teacher_output / "checkpoints" / cfg.teacher_tag
    for source in cfg.clients:
        checkpoint = checkpoint_dir / f"{source}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing CE teacher {checkpoint}; complete the matching CE cell first"
            )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CEModel(feature_dim=256, pretrained=False)
        model.load_state_dict(saved["model"])
        teachers.append(TeacherAdapter(model).to(device).eval())
        metas.append(saved.get("meta", {"source": source}))
    return teachers, metas


def co_boost_kd_epoch(
    synthesizer: COBOOSTSynthesizer,
    student: CEModel,
    teacher: WEnsemble,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    ods_eta: float,
    device: torch.device,
) -> float:
    """Official ODS + KL update, with binary-safe metric bookkeeping."""
    student.train()
    teacher.eval()
    total = 0.0
    batches = 0
    for images, _ in synthesizer.get_data(labeled=True):
        images = images.to(device, non_blocking=True).requires_grad_(True)
        teacher_logits = teacher(images)
        random_direction = torch.empty_like(teacher_logits).uniform_(-1.0, 1.0)
        ods_objective = (
            random_direction * F.softmax(teacher_logits / 4.0, dim=1)
        ).sum()
        image_gradient = torch.autograd.grad(
            ods_objective, images, only_inputs=True
        )[0]
        perturbed = (images + ods_eta * image_gradient.sign()).detach()

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(perturbed)
        with torch.no_grad():
            teacher_logits = teacher(perturbed)
        loss = criterion(student_logits, teacher_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def model_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, np.ndarray]:
    model.to(device).eval()
    logits = []
    labels = []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher_output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--synth_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--g_steps", type=int, default=30)
    parser.add_argument("--kd_lr", type=float, default=0.01)
    parser.add_argument("--lr_g", type=float, default=1e-3)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--heldout", choices=["", *SOURCES], default="")
    parser.add_argument("--limit_test", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(
        args.seed, args.epochs, args.image_size, args.synth_size, args.heldout
    )
    result_path = args.output / "results" / f"{cfg.tag}.json"
    if result_path.exists():
        print(f"SKIP {result_path}")
        return
    teacher_output = args.teacher_output or args.output

    seed_everything(cfg.seed)
    frames = load_sources(args.data_root)
    audit = {
        source: {
            split: {
                "n": len(rows := subset_rows(frame, split, 0, cfg.seed)),
                "positive": int(rows["_label"].sum()),
                "patients": int(rows["_patient"].nunique()),
            }
            for split in ("train", "val", "test")
        }
        for source, frame in frames.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Co-Boosting requires a CUDA/HIP GPU")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    teachers, teacher_meta = load_teachers(cfg, teacher_output, device)
    initial_weights = torch.full(
        (len(teachers), 1), 1.0 / len(teachers), device=device
    )
    teacher = WEnsemble(teachers, initial_weights).to(device).eval()
    student = CEModel(feature_dim=256, pretrained=False).to(device)

    run_dir = args.output / "coboost_synthesis" / cfg.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.synth_size, cfg.synth_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    normalizer = datafree.utils.Normalizer(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    generator = Generator(
        nz=256, ngf=64, img_size=cfg.synth_size, nc=3
    ).to(device)
    method_args = SimpleNamespace(
        hs=1.0,
        print_freq=max(1, args.g_steps // 3),
        weighted=True,
        wa_steps=1,
        mu=0.01,
        wdc=0.99,
        his=True,
        batchonly=False,
        batchused=False,
        logger=PrintLogger(),
    )
    synthesizer = COBOOSTSynthesizer(
        teacher=teacher,
        mdl_list=teachers,
        student=student,
        generator=generator,
        nz=256,
        num_classes=2,
        img_size=(3, cfg.synth_size, cfg.synth_size),
        save_dir=str(run_dir),
        iterations=args.g_steps,
        lr_g=args.lr_g,
        synthesis_batch_size=args.batch_size,
        sample_batch_size=args.batch_size,
        adv=1.0,
        bn=0.0,
        oh=1.0,
        criterion=datafree.criterions.KLDiv(T=1.0),
        transform=transform,
        normalizer=normalizer,
        autocast=datafree.utils.dummy_ctx,
        args=method_args,
    )

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.kd_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    criterion = datafree.criterions.KLDiv(T=args.kd_temperature)
    last_checkpoint = (
        args.output / "checkpoints" / cfg.tag / "student.last.pt"
    )
    start_epoch = 0
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        student.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        teacher.mdl_w_list = saved["teacher_weights"].to(device)
        start_epoch = int(saved["epoch"])
        print(f"resume at epoch {start_epoch}/{cfg.epochs}", flush=True)

    for epoch in range(start_epoch, cfg.epochs):
        method_args.current_epoch = epoch
        synthesizer.synthesize(cur_ep=epoch)
        teacher = synthesizer.teacher.to(device)
        loss = co_boost_kd_epoch(
            synthesizer,
            student,
            teacher,
            criterion,
            optimizer,
            ods_eta=8.0 / 255.0,
            device=device,
        )
        scheduler.step()
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            weights = teacher.mdl_w_list.detach().view(-1).cpu().tolist()
            print(
                f"epoch {epoch+1}/{cfg.epochs} kd={loss:.4f} "
                f"weights={[round(x, 4) for x in weights]}",
                flush=True,
            )
        if (epoch + 1) % args.save_every == 0 and epoch + 1 < cfg.epochs:
            last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                {
                    "model": state_to_fp16_cpu(student.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "teacher_weights": teacher.mdl_w_list.detach().cpu(),
                    "epoch": epoch + 1,
                },
                last_checkpoint,
            )

    final_checkpoint = args.output / "checkpoints" / cfg.tag / "student.pt"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "model": state_to_fp16_cpu(student.state_dict()),
            "teacher_weights": teacher.mdl_w_list.detach().cpu(),
            "epoch": cfg.epochs,
        },
        final_checkpoint,
    )
    if last_checkpoint.exists():
        last_checkpoint.unlink()

    test_sets: Dict[str, Dataset] = {}
    for source, frame in frames.items():
        rows = subset_rows(frame, "test", args.limit_test, cfg.seed)
        test_sets[source] = FundusDataset(
            rows, build_transform(cfg.image_size, False)
        )
    test_sets["pooled"] = ConcatDataset(list(test_sets.values()))

    evaluations = {}
    for domain, dataset in test_sets.items():
        loader = make_loader(dataset, max(64, args.batch_size), args.workers)
        logits, labels = model_logits(student, loader, device)
        evaluations[domain] = {
            "coboost_student": binary_metrics(logits, labels)
        }

    primary = "coboost_student"
    worst_domain = min(
        evaluations[source][primary]["balanced_accuracy"]
        for source in cfg.clients
    )
    result = {
        "schema_version": 1,
        "method": "Co-Boosting",
        "method_source": "official ICLR 2024 synthesizer and KD objective",
        "adaptation": (
            f"CE teachers from matched runs; synthetic resolution {cfg.synth_size}"
        ),
        "cell": asdict(cfg),
        "tag": cfg.tag,
        "code_revision": git_revision(),
        "data_audit": audit,
        "clients": cfg.clients,
        "teacher_meta": teacher_meta,
        "teacher_weights": teacher.mdl_w_list.detach().view(-1).cpu().tolist(),
        "evaluation": evaluations,
        "primary_method": primary,
        "worst_participating_domain_balanced_accuracy": worst_domain,
        "elapsed_s": time.time() - start,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    atomic_json_dump(result, result_path)
    atomic_json_dump(
        {
            "status": "complete",
            "tag": cfg.tag,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "result": str(result_path),
        },
        args.output / "results" / f"{cfg.tag}.meta.json",
    )
    print(
        f"COMPLETE {cfg.tag}: pooled BA="
        f"{evaluations['pooled'][primary]['balanced_accuracy']*100:.2f}% "
        f"elapsed={(time.time()-start)/60:.1f}min",
        flush=True,
    )


if __name__ == "__main__":
    main()
