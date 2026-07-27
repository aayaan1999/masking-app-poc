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
    # Draw a solid filled panel only — remove the asterisk overlay so the
    # redaction is a plain black box (keeps original pixels overwritten).
    draw.rectangle(bbox, fill=FILL_COLOR)


def apply_redactions(page_image, instances):
    """instances: list of {"bbox": (l,t,r,b), ...} for THIS page only."""
    masked = page_image.copy()
    draw = ImageDraw.Draw(masked)
    for inst in instances:
        draw_redaction(draw, inst["bbox"])
    return masked
