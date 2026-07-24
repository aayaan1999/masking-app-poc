"""
engine/masking.py
Draws the actual redaction. The box is still solid (the original pixels
underneath are fully overwritten — this remains a real redaction, not a
translucent overlay), but instead of a bare black bar it's filled with a
dark panel and a centered "********" pattern, matching how redaction
looks in most document-masking tools.
"""

from PIL import ImageDraw, ImageFont

FILL_COLOR = (20, 20, 20)      # near-black panel
TEXT_COLOR = (235, 235, 235)   # light asterisks on top
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
MASK_CHAR = "*"

_font_cache = {}


def _font(size):
    size = max(6, size)
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_redaction(draw: ImageDraw.ImageDraw, bbox):
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return
    draw.rectangle(bbox, fill=FILL_COLOR)

    # Size the asterisk text to the box height, then repeat "*" until it
    # roughly fills the width, so a wide field (address) and a narrow one
    # (a 4-digit PIN) both read clearly as redacted.
    font_size = max(8, int(h * 0.6))
    font = _font(font_size)
    one_char_w = draw.textlength(MASK_CHAR, font=font) or (font_size * 0.6)
    n_chars = max(3, int(w / max(1, one_char_w * 1.3)))
    pattern = MASK_CHAR * n_chars

    text_w = draw.textlength(pattern, font=font)
    while text_w > w * 0.94 and n_chars > 1:
        n_chars -= 1
        pattern = MASK_CHAR * n_chars
        text_w = draw.textlength(pattern, font=font)

    tx = left + (w - text_w) / 2
    ty = top + (h - font_size) / 2
    draw.text((tx, ty), pattern, fill=TEXT_COLOR, font=font)


def apply_redactions(page_image, instances):
    """instances: list of {"bbox": (l,t,r,b), ...} for THIS page only."""
    masked = page_image.copy()
    draw = ImageDraw.Draw(masked)
    for inst in instances:
        draw_redaction(draw, inst["bbox"])
    return masked
