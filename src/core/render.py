"""
模块0: 汉字数据生成与预处理
功能：字体渲染 → 灰度图 → 二值化 → 居中裁剪 → resize
输出：64×64 或 128×128 二值汉字图
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class CharRenderer:
    """汉字渲染器：根据输入汉字生成二值图"""

    def __init__(
        self,
        font_path: str,
        font_size: int = 200,
        target_size: int = 64,
        threshold: int = 128,
        invert: bool = True,
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.target_size = target_size
        self.threshold = threshold
        self.invert = invert

    def render(self, char: str) -> Image.Image:
        if len(char) != 1:
            raise ValueError(f"Input must be a single char, got: '{char}'")

        font = ImageFont.truetype(self.font_path, self.font_size)
        canvas_size = int(self.font_size * 1.5)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        offset_x = bbox[0]
        offset_y = bbox[1]

        x = (canvas_size - text_w) // 2 - offset_x
        y = (canvas_size - text_h) // 2 - offset_y
        draw.text((x, y), char, fill=0, font=font)

        # Binarize
        img_arr = np.array(img)
        binary = (img_arr < self.threshold).astype(np.uint8) * 255
        img_binary = Image.fromarray(binary.astype(np.uint8))

        # Center crop
        img_cropped = self._center_crop(img_binary)

        # Resize
        img_resized = img_cropped.resize(
            (self.target_size, self.target_size), Image.LANCZOS
        )

        # Re-threshold
        img_final = self._re_threshold(img_resized)
        return img_final

    def _center_crop(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)

        if self.invert:
            fg_mask = arr > self.threshold
            bg_color = 0
        else:
            fg_mask = arr < self.threshold
            bg_color = 255

        coords = np.argwhere(fg_mask)
        if len(coords) == 0:
            return img

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))

        w, h = cropped.size
        max_side = max(w, h)
        pad_img = Image.new("L", (max_side, max_side), color=bg_color)

        offset_x = (max_side - w) // 2
        offset_y = (max_side - h) // 2
        pad_img.paste(cropped, (offset_x, offset_y))

        return pad_img

    def _re_threshold(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        binary = (arr >= 128).astype(np.uint8) * 255
        return Image.fromarray(binary.astype(np.uint8))


def load_chars_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = []
    seen = set()
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" and ch not in seen and ch.strip():
            chars.append(ch)
            seen.add(ch)
    return chars


def batch_render(
    chars: list[str],
    font_path: str,
    output_dir: str,
    target_size: int = 64,
    font_size: int = 200,
    threshold: int = 128,
    invert: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    renderer = CharRenderer(
        font_path=font_path,
        font_size=font_size,
        target_size=target_size,
        threshold=threshold,
        invert=invert,
    )

    success = 0
    fail = 0
    for char in chars:
        try:
            img = renderer.render(char)
            code = ord(char)
            filename = f"{char}_U+{code:04X}.png"
            img.save(os.path.join(output_dir, filename))
            success += 1
        except Exception as e:
            print(f"[ERROR] '{char}': {e}")
            fail += 1

    print(f"Done: success={success}, fail={fail}")
    return success, fail


if __name__ == "__main__":
    raise SystemExit("请使用: python src/scripts/render.py --font-path <字体文件>")
