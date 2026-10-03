#!/usr/bin/env python3
"""
gui.py — ابزار هم‌ترازی دو تصویر برای پروژه‌ی تصاویر ترکیبی (Hybrid Images)

نحوه‌ی اجرا:
    python gui.py path/to/image1.jpg path/to/image2.jpg

گزینه‌های بیشتر:
    python gui.py --help
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    """خواندن تصویر از مسیر داده‌شده و تبدیل آن به آرایه‌ی NumPy (RGB)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def pick_points(img1, img2, n_points=3):
    """
    نمایش دو تصویر کنار هم و دریافت n_points جفت نقطه‌ی متناظر.
    ترتیب کلیک: یک‌بار روی تصویر ۱، سپس همان نقطه روی تصویر ۲، و تکرار.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1)
    axes[0].set_title(f"تصویر ۱ — 0/{n_points} نقطه")
    axes[0].axis("off")
    axes[1].imshow(img2)
    axes[1].set_title(f"تصویر ۲ — 0/{n_points} نقطه")
    axes[1].axis("off")
    fig.suptitle(
        "ابتدا یک نقطه روی تصویر ۱ کلیک کنید، سپس همان نقطه را روی تصویر ۲ مشخص کنید.\n"
        f"این کار را برای {n_points} جفت نقطه (مثلاً چشم چپ، چشم راست، نوک بینی) تکرار کنید."
    )

    pts1, pts2 = [], []

    def onclick(event):
        if event.inaxes == axes[0] and len(pts1) == len(pts2) and len(pts1) < n_points:
            pts1.append((event.xdata, event.ydata))
            axes[0].plot(event.xdata, event.ydata, "r+", markersize=14, markeredgewidth=2)
            axes[0].set_title(f"تصویر ۱ — {len(pts1)}/{n_points} نقطه")
            fig.canvas.draw()
        elif event.inaxes == axes[1] and len(pts2) < len(pts1):
            pts2.append((event.xdata, event.ydata))
            axes[1].plot(event.xdata, event.ydata, "b+", markersize=14, markeredgewidth=2)
            axes[1].set_title(f"تصویر ۲ — {len(pts2)}/{n_points} نقطه")
            fig.canvas.draw()

        if len(pts1) == n_points and len(pts2) == n_points:
            fig.suptitle("تمام نقاط ثبت شد — در حال بستن پنجره...")
            fig.canvas.draw()
            plt.pause(0.4)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

    if len(pts1) < n_points or len(pts2) < n_points:
        raise RuntimeError(
            "تعداد نقاط کافی انتخاب نشد. برنامه را دوباره اجرا کرده و همه‌ی نقاط را مشخص کنید."
        )

    return np.array(pts1, dtype=np.float64), np.array(pts2, dtype=np.float64)


def compute_affine(src_pts, dst_pts):
    """
    محاسبه‌ی ماتریس تبدیل آفین (۳×۳ به‌صورت همگن) که src_pts را به dst_pts می‌برد.
    مدل: [x', y', 1]^T = M @ [x, y, 1]^T
    اگر بیش از ۳ جفت نقطه داده شود، از حداقل مربعات استفاده می‌شود.
    """
    n = src_pts.shape[0]
    if n < 3:
        raise ValueError("برای محاسبه‌ی تبدیل آفین حداقل به ۳ جفت نقطه نیاز است.")

    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = xp
        b[2 * i + 1] = yp

    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, c, e, bb, d, f = params[0], params[1], params[2], params[3], params[4], params[5]
    M = np.array([[a, c, e],
                  [bb, d, f],
                  [0, 0, 1]])
    return M


def warp_image(img, M, output_shape):
    """
    اعمال تبدیل آفین M روی img و بازگرداندن تصویری با اندازه‌ی output_shape (ارتفاع، عرض).
    از نگاشت معکوس (inverse mapping) با نزدیک‌ترین همسایه استفاده می‌شود.
    """
    h_out, w_out = output_shape
    M_inv = np.linalg.inv(M)
    out = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

    ys, xs = np.meshgrid(np.arange(h_out), np.arange(w_out), indexing="ij")
    ones = np.ones_like(xs)
    dst_coords = np.stack([xs.ravel(), ys.ravel(), ones.ravel()])
    src_coords = M_inv @ dst_coords

    src_x = np.round(src_coords[0]).astype(int).reshape(h_out, w_out)
    src_y = np.round(src_coords[1]).astype(int).reshape(h_out, w_out)

    valid = (
        (src_x >= 0) & (src_x < img.shape[1]) &
        (src_y >= 0) & (src_y < img.shape[0])
    )
    out[valid] = img[src_y[valid], src_x[valid]]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="ابزار هم‌ترازی دو تصویر برای ساخت تصویر ترکیبی (Hybrid Image)."
    )
    parser.add_argument("image1", help="مسیر تصویر اول (پایه‌ی هم‌ترازی؛ خروجی left.png)")
    parser.add_argument("image2", help="مسیر تصویر دوم (هم‌تراز می‌شود؛ خروجی right.png)")
    parser.add_argument(
        "-n", "--num-points", type=int, default=3,
        help="تعداد نقاط متناظر برای هم‌ترازی (پیش‌فرض: ۳؛ حداقل ۳)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=".",
        help="پوشه‌ی خروجی برای ذخیره‌ی left.png و right.png (پیش‌فرض: پوشه‌ی جاری)",
    )
    parser.add_argument(
        "-t", "--correspondence", default=None,
        help="مسیر فایل JSON نقاط متناظر؛ اگر موجود باشد خوانده می‌شود، وگرنه بعد از کلیک‌کردن ساخته می‌شود.",
    )
    args = parser.parse_args()

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    if args.correspondence and os.path.exists(args.correspondence):
        with open(args.correspondence, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts1 = np.array(data["points1"], dtype=np.float64)
        pts2 = np.array(data["points2"], dtype=np.float64)
        print(f"نقاط متناظر از فایل «{args.correspondence}» بارگذاری شد.")
    else:
        pts1, pts2 = pick_points(img1, img2, n_points=args.num_points)
        corr_path = args.correspondence or os.path.join(args.output_dir, "correspondence.json")
        os.makedirs(os.path.dirname(corr_path) or ".", exist_ok=True)
        with open(corr_path, "w", encoding="utf-8") as f:
            json.dump({"points1": pts1.tolist(), "points2": pts2.tolist()}, f, indent=2, ensure_ascii=False)
        print(f"نقاط متناظر در «{corr_path}» ذخیره شد (برای اجرای بعدی می‌توانید با -t دوباره استفاده کنید).")

    # محاسبه‌ی تبدیلی که تصویر ۲ را به سیستم مختصات تصویر ۱ می‌برد
    M = compute_affine(pts2, pts1)
    out_shape = (img1.shape[0], img1.shape[1])
    img2_aligned = warp_image(img2, M, out_shape)

    os.makedirs(args.output_dir, exist_ok=True)
    left_path = os.path.join(args.output_dir, "left.png")
    right_path = os.path.join(args.output_dir, "right.png")
    Image.fromarray(img1).save(left_path)
    Image.fromarray(img2_aligned).save(right_path)
    print(f"ذخیره شد: {left_path}")
    print(f"ذخیره شد: {right_path}")

    # نمایش نتیجه‌ی نهایی برای بررسی بصری هم‌ترازی
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title("left.png")
    axes[0].axis("off")
    axes[1].imshow(img2_aligned)
    axes[1].set_title("right.png (هم‌تراز‌شده)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
