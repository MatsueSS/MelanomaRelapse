import os
import sys
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import pyvips
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

# ==================== НАСТРОЙКИ ====================
URL = "https://oos.eu-west-2.outscale.com/hdh-datachallenge/Visiomel/breslow_empty.zip"

WORK_DIR  = Path.home() / "visiomel_work"
TIFF_DIR  = WORK_DIR / "tiff"
NPZ_DIR   = WORK_DIR / "embeddings"
LOG_FILE  = WORK_DIR / "process.log"
STATUS_DIR = WORK_DIR / "status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

def status_path(stem):
    return STATUS_DIR / f"{stem}.json"

def load_status(stem):
    p = status_path(stem)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_status(stem, data):
    p = status_path(stem)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)   # атомарная запись



PATCH      = 224
STRIDE     = 224
BATCH      = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_EVERY = 20          # чаще = надёжнее, но больше I/O
USE_AMP    = (DEVICE == "cuda")
MAX_RETRY  = 3
RETRY_WAIT = 30          # секунд между попытками
# ====================================================


TIFF_DIR.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)


# ---------- логирование ----------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fmt(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    if h: return f"{h}ч {m}м {s}с"
    if m: return f"{m}м {s}с"
    return f"{s}с"


# ---------- UNI ----------
_uni_cache = {"model": None, "transform": None}


def load_uni():
    if _uni_cache["model"] is not None:
        return _uni_cache["model"], _uni_cache["transform"]

    log("Загружаю UNI...")
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    model = model.to(DEVICE).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    _uni_cache["model"] = model
    _uni_cache["transform"] = transform
    log(f"UNI на {DEVICE}, AMP={'on' if USE_AMP else 'off'}, BATCH={BATCH}")
    return model, transform


# ---------- обработка одного TIFF ----------
# Флаг для аварийного сохранения чекпоинта
_emergency = {"embs": None, "coords": None, "done": 0, "ckpt_path": None}


def _emergency_save(signum, frame):
    """Сохраняет чекпоинт при SIGTERM/SIGINT."""
    if _emergency["embs"] and _emergency["ckpt_path"]:
        try:
            arr = np.concatenate(_emergency["embs"], axis=0)
            np.savez(
                _emergency["ckpt_path"],
                embeddings=arr,
                coords=np.array(_emergency["coords"]),
                done=_emergency["done"],
            )
            log(f"  [аварийный чекпоинт: {_emergency['done']} патчей]")
        except Exception as e:
            log(f"  [ошибка аварийного сохранения: {e}]")
    sys.exit(0)


def encode_tiff(tiff_path, npz_path):
    """Прогоняет TIFF через UNI с чекпоинтами."""
    model, transform = load_uni()

    tiff_path = Path(tiff_path)
    npz_path = Path(npz_path)
    ckpt_path = npz_path.parent / (npz_path.stem + "_ckpt.npz")

    t0 = time.time()
    log(f"  Открываю {tiff_path.name}...")
    img = pyvips.Image.new_from_file(str(tiff_path), access="sequential")
    if img.interpretation == "ycbcr":
        img = img.colourspace("srgb")

    w, h = img.width, img.height
    n_x = (w + STRIDE - 1) // STRIDE
    n_y = (h + STRIDE - 1) // STRIDE
    total_patches = n_x * n_y
    total_batches = (total_patches + BATCH - 1) // BATCH

    log(f"  Размер: {w}x{h}, патчей: {total_patches}, батчей: {total_batches}")

    # ---------- resume ----------
    start_patch = 0
    embs = []
    coords = []

    if ckpt_path.exists():
        try:
            ckpt = np.load(ckpt_path)
            embs = [ckpt["embeddings"]]
            coords = list(map(tuple, ckpt["coords"]))
            start_patch = int(ckpt["done"])
            log(f"  Возобновляю с патча {start_patch}/{total_patches}")
        except Exception as e:
            log(f"  Чекпоинт битый, начинаю с нуля: {e}")
            embs, coords, start_patch = [], [], 0

    # регистрируем аварийное сохранение
    _emergency["embs"] = embs
    _emergency["coords"] = coords
    _emergency["done"] = start_patch
    _emergency["ckpt_path"] = ckpt_path
    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT, _emergency_save)

    buf = []
    done_patches = start_patch
    done_batches = start_patch // BATCH
    skipped = 0
    t_start = time.time()

    with torch.inference_mode():
        for y in range(0, h, STRIDE):
            for x in range(0, w, STRIDE):
                # пропускаем уже обработанные
                if skipped < start_patch:
                    skipped += 1
                    continue

                patch = img.crop(x, y, min(PATCH, w - x), min(PATCH, h - y))
                arr = np.ndarray(
                    buffer=patch.write_to_memory(),
                    dtype=np.uint8,
                    shape=[patch.height, patch.width, patch.bands],
                )

                if arr.shape[0] != PATCH or arr.shape[1] != PATCH:
                    padded = np.zeros((PATCH, PATCH, 3), dtype=np.uint8)
                    ch = min(3, arr.shape[2])
                    padded[:arr.shape[0], :arr.shape[1], :ch] = arr[..., :ch]
                    arr = padded
                elif arr.shape[2] > 3:
                    arr = arr[..., :3]

                buf.append(transform(Image.fromarray(arr)))
                coords.append((x, y))
                done_patches += 1

                if len(buf) == BATCH:
                    batch = torch.stack(buf).to(DEVICE)
                    buf = []

                    if USE_AMP:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(batch)
                    else:
                        out = model(batch)

                    embs.append(out.float().cpu().numpy())
                    done_batches += 1

                    if done_batches % 10 == 0:
                        el = time.time() - t_start
                        rate = (done_patches - start_patch) / el if el > 0 else 0
                        remaining = total_patches - done_patches
                        eta = remaining / rate if rate > 0 else 0
                        pct = 100 * done_patches / total_patches
                        log(
                            f"  [{pct:5.1f}%] батч {done_batches}/{total_batches} | "
                            f"патчей {done_patches}/{total_patches} | "
                            f"прошло {fmt(el)} | осталось ~{fmt(eta)} | "
                            f"{rate:.1f} патчей/с"
                        )

                    if done_batches % CKPT_EVERY == 0:
                        ckpt_embs = np.concatenate(embs, axis=0)
                        np.savez(
                            ckpt_path,
                            embeddings=ckpt_embs,
                            coords=np.array(coords),
                            done=done_patches,
                        )
                        _emergency["embs"] = embs
                        _emergency["coords"] = coords
                        _emergency["done"] = done_patches
                        log(f"  [чекпоинт: {done_patches} патчей]")

        # хвост
        if buf:
            batch = torch.stack(buf).to(DEVICE)
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch)
            else:
                out = model(batch)
            embs.append(out.float().cpu().numpy())

    embs = np.concatenate(embs, axis=0)
    coords = np.array(coords)

    np.savez(npz_path, embeddings=embs, coords=coords)

    # удаляем чекпоинт после успешного финала
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = time.time() - t0
    log(f"  ГОТОВО: {embs.shape}, время {fmt(elapsed)}")
    return embs.shape


# ---------- работа с ZIP ----------
import re

def get_zip_listing():
    """Получает список файлов внутри ZIP."""
    result = subprocess.run(
        ["cloud_unzip", "-l", URL],
        capture_output=True, text=True, timeout=600,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    files = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not line.lower().endswith((".tif", ".tiff", ")")):
            # пропускаем заголовок "Files in the ZIP archive (49):"
            if not re.search(r"\.tiff?\s+\(", line.lower()):
                continue

        # Отрезаем суффикс " (2.09 GB)"
        line = re.sub(r"\s+\([\d.]+\s*[KMGT]?B\)\s*$", "", line)
        line = line.strip()

        if line.lower().endswith((".tif", ".tiff")):
            files.append(line)

    return files

def download_tiff(member, out_path, retries=MAX_RETRY):
    """Скачивает TIFF с retry."""
    out_path = Path(out_path)
    for attempt in range(1, retries + 1):
        try:
            # ИСПРАВЛЕНО: используем -o и указываем ПОЛНЫЙ путь к файлу
            subprocess.run(
                ["cloud_unzip", "-e", member, "-o", str(out_path), URL],
                check=True, timeout=1800,
                capture_output=True, text=True,
            )
            # Проверяем, что файл действительно создан
            if not out_path.exists():
                raise FileNotFoundError(f"cloud_unzip не создал {out_path}")
            return
        except subprocess.CalledProcessError as e:
            log(f"  попытка {attempt}/{retries} не удалась: {e.stderr[:200] if e.stderr else e}")
            if attempt < retries:
                time.sleep(RETRY_WAIT)
        except Exception as e:
            log(f"  попытка {attempt}/{retries} не удалась: {e}")
            if attempt < retries:
                time.sleep(RETRY_WAIT)
    raise RuntimeError(f"не удалось скачать {member} после {retries} попыток")


def main():
    log("=" * 60)
    log("СТАРТ")
    log(f"DEVICE={DEVICE}, BATCH={BATCH}, AMP={USE_AMP}")

    files = get_zip_listing()
    log(f"В ZIP найдено файлов: {len(files)}")
    if not files:
        log("Пустой список — проверь cloud_unzip -l")
        return

    for i, member in enumerate(files, 1):
        fname = os.path.basename(member)
        stem = os.path.splitext(fname)[0]
        npz_path = NPZ_DIR / f"{stem}_uni.npz"
        tiff_path = TIFF_DIR / fname

        if npz_path.exists():
            log(f"[{i}/{len(files)}] SKIP {fname} (готово)")
            continue

        log(f"[{i}/{len(files)}] START {fname}")

        # скачивание
        try:
            if not tiff_path.exists():
                download_tiff(member, tiff_path)
            size_mb = tiff_path.stat().st_size / 1e6
            log(f"  скачан {fname} ({size_mb:.1f} МБ)")
        except Exception as e:
            log(f"  ОШИБКА скачивания: {e}")
            continue

        # обработка
        try:
            encode_tiff(tiff_path, npz_path)
        except Exception as e:
            log(f"  ОШИБКА обработки: {e}")
            # TIFF НЕ удаляем — попробуем в следующий запуск
            continue

        # удаление TIFF
        try:
            tiff_path.unlink()
            log(f"  удалён {fname}")
        except Exception as e:
            log(f"  не смог удалить {fname}: {e}")

    log("ВСЁ ГОТОВО")
    log("=" * 60)


if __name__ == "__main__":
    main()
