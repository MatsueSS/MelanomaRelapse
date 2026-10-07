import os
import sys
import signal
import subprocess
import time
import re
from pathlib import Path

# Сбрасываем прокси ДО импорта timm/huggingface_hub
for key in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'https_proxy',
            'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY']:
    os.environ.pop(key, None)

# Дополнительно для httpx (он читает переменные в момент импорта)
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

import numpy as np
import torch
import pyvips
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

# ==================== НАСТРОЙКИ ====================
URL = "https://oos.eu-west-2.outscale.com/hdh-datachallenge/Visiomel/breslow_empty.zip"

WORK_DIR   = Path.home() / "visiomel_work"
TIFF_DIR   = WORK_DIR / "tiff"
NPZ_DIR    = WORK_DIR / "embeddings"
STATUS_DIR = WORK_DIR / "status"
LOG_FILE   = WORK_DIR / "process.log"

PATCH      = 224
STRIDE     = 224
BATCH      = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_EVERY = 20
USE_AMP    = (DEVICE == "cuda")
MAX_RETRY  = 5
RETRY_WAIT = 30
# ====================================================


for d in (WORK_DIR, TIFF_DIR, NPZ_DIR, STATUS_DIR):
    d.mkdir(parents=True, exist_ok=True)


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


# ---------- статус: просто 0 или 1 ----------
def status_file(stem):
    return STATUS_DIR / f"{stem}.downloaded"


def is_downloaded(stem):
    """True, если флаг == '1'."""
    f = status_file(stem)
    return f.exists() and f.read_text().strip() == "1"


def mark_downloaded(stem):
    """Ставит флаг '1' (скачан полностью)."""
    f = status_file(stem)
    tmp = f.with_suffix(".tmp")
    tmp.write_text("1")
    tmp.replace(f)   # атомарная замена


def mark_not_downloaded(stem):
    """Ставит флаг '0' (или удаляет старый '1' — значит файл надо качать заново)."""
    f = status_file(stem)
    tmp = f.with_suffix(".tmp")
    tmp.write_text("0")
    tmp.replace(f)


# ---------- UNI ----------
_uni_cache = {"model": None, "transform": None}


def load_uni():
    if _uni_cache["model"] is not None:
        return _uni_cache["model"], _uni_cache["transform"]
    log("Загружаю UNI...")
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni", pretrained=True,
        init_values=1e-5, dynamic_img_size=True,
    )
    model = model.to(DEVICE).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    _uni_cache["model"] = model
    _uni_cache["transform"] = transform
    log(f"UNI на {DEVICE}, AMP={'on' if USE_AMP else 'off'}, BATCH={BATCH}")
    return model, transform


# ---------- аварийный чекпоинт энкодинга ----------
_emergency = {"embs": None, "coords": None, "done": 0, "ckpt_path": None}


def _emergency_save(signum, frame):
    if _emergency["embs"] and _emergency["ckpt_path"]:
        try:
            arr = np.concatenate(_emergency["embs"], axis=0)
            np.savez(_emergency["ckpt_path"],
                     embeddings=arr,
                     coords=np.array(_emergency["coords"]),
                     done=_emergency["done"])
            log(f"  [аварийный чекпоинт: {_emergency['done']} патчей]")
        except Exception as e:
            log(f"  [ошибка аварийного сохранения: {e}]")
    sys.exit(0)


def encode_tiff(tiff_path, npz_path):
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
            log(f"  Чекпоинт битый: {e}, начинаю с нуля")
            embs, coords, start_patch = [], [], 0

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
                if skipped < start_patch:
                    skipped += 1
                    continue
                patch = img.crop(x, y, min(PATCH, w - x), min(PATCH, h - y))
                arr = np.ndarray(buffer=patch.write_to_memory(), dtype=np.uint8,
                                 shape=[patch.height, patch.width, patch.bands])
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
                        log(f"  [{pct:5.1f}%] батч {done_batches}/{total_batches} | "
                            f"патчей {done_patches}/{total_patches} | "
                            f"прошло {fmt(el)} | осталось ~{fmt(eta)} | "
                            f"{rate:.1f} патчей/с")

                    if done_batches % CKPT_EVERY == 0:
                        ckpt_embs = np.concatenate(embs, axis=0)
                        np.savez(ckpt_path, embeddings=ckpt_embs,
                                 coords=np.array(coords), done=done_patches)
                        _emergency["embs"] = embs
                        _emergency["coords"] = coords
                        _emergency["done"] = done_patches
                        log(f"  [чекпоинт: {done_patches} патчей]")

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

    # атомарная запись .npz
    tmp_npz = npz_path.with_suffix(".npz.tmp")
    np.savez(tmp_npz, embeddings=embs, coords=coords)
    tmp_npz.replace(npz_path)

    if ckpt_path.exists():
        ckpt_path.unlink()

    log(f"  ГОТОВО: {embs.shape}, время {fmt(time.time() - t0)}")


# ---------- работа с ZIP ----------
def get_zip_listing():
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
        # отрезаем "(2.09 GB)" в конце
        line = re.sub(r"\s+\([\d.]+\s*[KMGT]?B\)\s*$", "", line)
        line = line.strip()
        if line.lower().endswith((".tif", ".tiff")):
            files.append(line)
    return files

def download_tiff(member, out_path, retries=MAX_RETRY):
    """Скачивает TIFF, перемещает в out_path, чистит вложенные mnt/..."""
    out_path = Path(out_path)
    tiff_dir = out_path.parent          # например, ~/visiomel_work/tiff

    for attempt in range(1, retries + 1):
        try:
            # cloud_unzip сохранит файл в tiff_dir/mnt/raid10/.../vs3zna6d.tif
            proc = subprocess.run(
                ["cloud_unzip", "-e", member, "-o", str(tiff_dir), URL],
                timeout=3600, capture_output=True, text=True,
            )
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

            ok = (proc.returncode == 0) and ("(100%)" in combined)
            if not ok:
                raise RuntimeError(
                    f"returncode={proc.returncode}, 100%={'yes' if '(100%)' in combined else 'no'}"
                )

            # Ищем, куда cloud_unzip реально положил файл
            fname = out_path.name
            candidates = list(tiff_dir.rglob(fname))
            if not candidates:
                raise FileNotFoundError(f"после распаковки не найден {fname}")

            extracted = candidates[0]

            # Перемещаем в корень tiff_dir
            if extracted != out_path:
                if out_path.exists():
                    out_path.unlink()
                extracted.rename(out_path)

            # Чистим пустые вложенные директории (mnt/raid10/...)
            _cleanup_empty_dirs(tiff_dir)

            return

        except Exception as e:
            log(f"  попытка {attempt}/{retries}: {e}")

        if out_path.exists():
            out_path.unlink()
        if attempt < retries:
            time.sleep(RETRY_WAIT)

    raise RuntimeError(f"не удалось скачать {member} после {retries} попыток")


def _cleanup_empty_dirs(root):
    """Удаляет пустые директории внутри root (например, mnt/raid10/...)."""
    # Идём снизу вверх: сначала самые глубокие
    for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
            except OSError:
                pass

FILES_TO_PROCESS = [
    "mnt/raid10/opendata_files/images_breslow_categ_final/breslow_empty/4qpgzvl7.tif",
    "mnt/raid10/opendata_files/images_breslow_categ_final/breslow_empty/23b543du.tif",
    "mnt/raid10/opendata_files/images_breslow_categ_final/breslow_empty/9v0bl8q7.tif",
    "mnt/raid10/opendata_files/images_breslow_categ_final/breslow_empty/f3fdygql.tif",
    "mnt/raid10/opendata_files/images_breslow_categ_final/breslow_empty/iib99c90.tif"
]

# ---------- главный цикл ----------
def main():
    log("=" * 60)
    log("СТАРТ")
    log(f"DEVICE={DEVICE}, BATCH={BATCH}, AMP={USE_AMP}")

    #files = get_zip_listing()
    files = FILES_TO_PROCESS
    log(f"В ZIP найдено файлов: {len(files)}")
    if not files:
        log("Пустой список")
        return

    for i, member in enumerate(files, 1):
        fname = os.path.basename(member)
        stem = os.path.splitext(fname)[0]
        npz_path = NPZ_DIR / f"{stem}_uni.npz"
        tiff_path = TIFF_DIR / fname

        # 1) Энкодинг уже сделан?
        if npz_path.exists():
            log(f"[{i}/{len(files)}] SKIP {fname} (уже заэнкожен)")
            # на всякий случай — почистим TIFF
            if tiff_path.exists():
                tiff_path.unlink()
            continue

        # 2) Флаг "скачан полностью" == 1?
        if not is_downloaded(stem):
            # Нет -> качаем заново целиком
            log(f"[{i}/{len(files)}] START {fname}")
            try:
                download_tiff(member, tiff_path)
                mark_downloaded(stem)   # ставим 1
                log(f"  скачан {fname} ({tiff_path.stat().st_size / 1e6:.1f} МБ), флаг=1")
            except Exception as e:
                mark_not_downloaded(stem)   # ставим 0
                log(f"  ОШИБКА скачивания: {e}")
                continue
        else:
            # Уже скачан ранее
            if not tiff_path.exists():
                # флаг стоит, а файла нет — сбрасываем флаг
                log(f"[{i}/{len(files)}] {fname}: флаг=1, но файла нет -> сбрасываю")
                mark_not_downloaded(stem)
                continue
            log(f"[{i}/{len(files)}] {fname}: флаг=1, файл на месте")

        # 3) Энкодинг
        try:
            encode_tiff(tiff_path, npz_path)
        except Exception as e:
            log(f"  ОШИБКА обработки: {e}")
            continue

        # 4) Удаление TIFF
        try:
            tiff_path.unlink()
            log(f"  удалён {fname}")
        except Exception as e:
            log(f"  не смог удалить {fname}: {e}")

    log("ВСЁ ГОТОВО")
    log("=" * 60)


if __name__ == "__main__":
    main()
