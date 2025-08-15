"""
PDF Image + Diagram Extractor Prototype
======================================

Purpose:
- Extract embedded raster images from a PDF
- Render pages at high resolution and detect vector diagrams as image regions
- Match images/regions to nearby captions (text blocks) using layout analysis
- Output an ordered list of image objects with metadata (bbox, caption, source)
- Save extracted images to disk

Dependencies:
- Requires: PyMuPDF (fitz), pdfplumber, opencv-python, numpy, Pillow, pytesseract, requests
  pip install pymupdf pdfplumber opencv-python numpy pillow pytesseract requests
- For OCR with pytesseract: install Tesseract OCR on your system and make sure it's in PATH.
  (Ubuntu: sudo apt-get install tesseract-ocr)

Usage example:
    python pdf_image_extractor.py --url "https://.../file.pdf" --outdir ./output

This is a prototype to demonstrate a robust hybrid extraction approach. Adjust thresholds
and heuristics for our document collection.

"""

import os
import re
import io
import json
import argparse
import requests
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import cv2
import pytesseract

# --------------------------- Utilities ---------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pil(img: Image.Image, path: str):
    img.save(path, format="PNG")


# --------------------------- Embedded image extraction ---------------------------

def extract_embedded_images(doc: fitz.Document, page_index: int, outdir: str):
    """
    Extract embedded raster images from a single page using PyMuPDF.
    Returns list of dicts with keys: bbox, path, width, height, source='embedded'
    """
    results = []
    page = doc[page_index]
    image_list = page.get_images(full=True)
    if not image_list:
        return results

    img_count = 0
    for img in image_list:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Filter out very small embedded images (likely icons/logos)
        if min(img_pil.width, img_pil.height) < 80 or (img_pil.width * img_pil.height) < 20000:
            continue

        # We don't have bbox for embedded images via this API directly.
        # Some heuristics later will match embedded images with rendered-region detection.
        fname = f"page_{page_index+1}_embedded_{img_count}.{ext}"
        outpath = os.path.join(outdir, fname)
        img_pil.save(outpath)

        results.append({
            "page": page_index + 1,
            "path": outpath,
            "width": img_pil.width,
            "height": img_pil.height,
            "source": "embedded",
            "xref": xref,
            # bbox unknown here; set to None and allow later spatial matching
            "bbox": None,
        })
        img_count += 1
    return results


# --------------------------- Render page and detect image-like regions ---------------------------

def render_page_to_image(doc: fitz.Document, page_index: int, zoom: int = 3):
    """Render page to PIL Image at `zoom` scale (integer). Returns PIL.Image and scale factor."""
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img, zoom


def detect_image_regions(pil_image: Image.Image, min_area: int = 50000, aggressive_page_merge: bool = True):
    """
    Use OpenCV to find rectangular regions likely to be diagrams/images.
    Returns list of bboxes in pixel coordinates: (x0,y0,x1,y1)
    Coordinates origin = top-left.
    """
    np_img = np.array(pil_image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Use adaptive thresholding to catch line-art and diagrams
    # Then find contours and filter by area and aspect ratio
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing and dilation to join fragmented components
    h_img, w_img = gray.shape
    k = max(9, int(min(h_img, w_img) * 0.02))
    if k % 2 == 0:
        k += 1  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    page_area = float(h_img * w_img)
    header_band = int(0.08 * h_img)  # exclude header/footer bands (~8% of page height)
    footer_band_y = h_img - header_band

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        # filter extreme thin shapes
        if w < 50 or h < 50:
            continue
        # filter near-full-page regions and extreme aspect ratios (likely full page or gutters)
        ar = w / float(h)
        if area / page_area > 0.9:
            continue
        if ar < 0.25 or ar > 6.0:
            continue
        # exclude header/footer repetitive areas (logos etc.)
        if (y + h) <= header_band or y >= footer_band_y:
            continue
        # optional: filter by solidity or extent if needed
        boxes.append((x, y, x + w, y + h))

    # Merge overlapping boxes (simple greedy merge)
    boxes = merge_boxes(boxes)
    # Merge boxes that are close to each other (bridge small gaps)
    boxes = merge_by_proximity(boxes, max_gap=int(min(h_img, w_img) * 0.03))

    # If many boxes cover a large portion of the page, consider the whole as one figure
    if aggressive_page_merge and len(boxes) >= 3:
        enc = enclosing_box(boxes)
        coverage = ((enc[2] - enc[0]) * (enc[3] - enc[1])) / page_area
        if coverage > 0.25:
            return [enc]

    return boxes


def merge_boxes(boxes, iou_thresh=0.2):
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        bx = boxes[i]
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            by = boxes[j]
            if iou(bx, by) > iou_thresh:
                # merge
                bx = [min(bx[0], by[0]), min(bx[1], by[1]), max(bx[2], by[2]), max(bx[3], by[3])]
                used[j] = True
        merged.append(tuple(bx))
    return merged


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union == 0:
        return 0
    return inter / union


# --- New helpers for better merging ---

def enclosing_box(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def boxes_touch_or_close(a, b, max_gap):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # distance between boxes if they don't overlap
    dx = max(max(ax0, bx0) - min(ax1, bx1), 0)
    dy = max(max(ay0, by0) - min(ay1, by1), 0)
    return dx <= max_gap and dy <= max_gap


def merge_by_proximity(boxes, max_gap=30):
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            a = boxes[i]
            merged = a
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                if boxes_touch_or_close(merged, b, max_gap):
                    merged = [min(merged[0], b[0]), min(merged[1], b[1]), max(merged[2], b[2]), max(merged[3], b[3])]
                    used[j] = True
                    changed = True
            used[i] = True
            new_boxes.append(merged)
        boxes = new_boxes
    return [tuple(b) for b in boxes]


# --------------------------- Layout analysis and caption matching ---------------------------

def extract_text_blocks_pdfplumber(pdf_path: str, page_number: int):
    """Return list of text blocks with bounding boxes using pdfplumber (1-indexed page_number)."""
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_number - 1 >= len(pdf.pages):
            return []
        page = pdf.pages[page_number - 1]
        # pdfplumber gives chars, lines, and grouped objects; use extract_words or extract_text with layout
        for b in page.extract_words(use_text_flow=True, keep_blank_chars=False, extra_attrs=["doctop", "top", "bottom", "x0", "x1", "y0", "y1", "size", "fontname", "upright"]):
            blocks.append({
                "text": b.get("text", "").strip(),
                "x0": b.get("x0"),
                "x1": b.get("x1"),
                "top": b.get("top"),
                "bottom": b.get("bottom"),
                "size": b.get("size"),
                "fontname": b.get("fontname"),
                "upright": b.get("upright"),
                # pdfplumber's vertical coords: 0 at top of page in points
            })
    # Optionally, merge nearby word-level boxes into blocks (not implemented here)
    return blocks


def match_caption_for_bbox(text_blocks, bbox_pts, page_height_pts, scale_factor):
    """
    Match nearest text block below or to the right/left of an image bbox.
    bbox_pts is in rendered-image pixel coords; convert to PDF points using scale_factor.
    Returns matched caption string or None.
    """
    # Convert bbox_pixels -> PDF points approximation: points = pixels / scale_factor
    x0_px, y0_px, x1_px, y1_px = bbox_pts
    x0_pt = x0_px / scale_factor
    x1_pt = x1_px / scale_factor
    y0_pt = y0_px / scale_factor
    y1_pt = y1_px / scale_factor

    # pdfplumber top coordinate is in points from top
    candidates = []
    for b in text_blocks:
        # Check vertical relation: prefer blocks that start below bbox bottom
        b_top = b.get("top")
        b_bottom = b.get("bottom")
        b_x0 = b.get("x0")
        b_x1 = b.get("x1")
        # considered if block is just below or overlapping vertically
        if b_top >= y1_pt - 5:
            # compute horizontal overlap
            horiz_overlap = max(0, min(b_x1, x1_pt) - max(b_x0, x0_pt))
            distance_y = b_top - y1_pt
            candidates.append((distance_y, -horiz_overlap, b.get("text")))

    if not candidates:
        # Try blocks that are to the right/left with small vertical distance
        for b in text_blocks:
            b_top = b.get("top")
            if abs(b_top - y0_pt) < 40:  # 40pt tolerance
                candidates.append((abs(b_top - y0_pt), 0, b.get("text")))

    if not candidates:
        return None

    # pick nearest candidate by distance and horizontal overlap
    candidates.sort(key=lambda x: (x[0], x[1]))
    caption = candidates[0][2]

    # Heuristic: if caption looks like a 'Figure' label, return it; else return truncated text
    if re.search(r"fig\.?|figure|diagram|table", caption, re.IGNORECASE):
        return caption
    # else return the short snippet up to 250 chars
    return caption[:250]


# --------------------------- OCR fallback ---------------------------

def ocr_image_region(pil_img: Image.Image):
    text = pytesseract.image_to_string(pil_img)
    return text.strip()


# --------------------------- Main pipeline per-page ---------------------------

def process_page(pdf_path: str, page_index: int, outdir: str, render_zoom: int = 3):
    """Process single page: extract embedded images, render page, detect regions, match captions."""
    ensure_dir(outdir)
    doc = fitz.open(pdf_path)
    # 1) extract embedded images
    embedded = extract_embedded_images(doc, page_index, outdir)

    # 2) render page and detect regions
    rendered_img, zoom = render_page_to_image(doc, page_index, zoom=render_zoom)
    rendered_path = os.path.join(outdir, f"page_{page_index+1}_rendered.png")
    save_pil(rendered_img, rendered_path)

    # Detect regions with stronger merging to avoid segmentation
    regions = detect_image_regions(rendered_img, min_area=50000)

    # 3) extract text blocks using pdfplumber for better coordinates
    text_blocks = extract_text_blocks_pdfplumber(pdf_path, page_index + 1)

    results = []

    # 4) handle embedded images first (they may lack bbox)
    for emb in embedded:
        # try to match embedded image to detected region via image content comparison
        # simple heuristic: compare sizes
        matched_region = None
        for r in regions:
            rx0, ry0, rx1, ry1 = r
            r_w, r_h = rx1 - rx0, ry1 - ry0
            # match if region size is similar to image pixel dims scaled by zoom
            if (abs(r_w - emb["width"]) < emb["width"] * 0.6) and (abs(r_h - emb["height"]) < emb["height"] * 0.6):
                matched_region = r
                break
        # If an embedded image is matched to a detected/merged region, we suppress the
        # standalone embedded output so that only the merged region crop is emitted.
        if matched_region:
            continue
        emb["bbox"] = None

        # try to find caption from nearby text using the (unknown) bbox -> skip
        emb["caption"] = None
        results.append(emb)

    # 5) handle detected regions (vector diagrams and others)
    region_count = 0
    for r in regions:
        # skip if already matched to an embedded image
        already = any([ent.get("bbox") == r and ent.get("source") == "embedded" for ent in results])
        if already:
            continue
        rx0, ry0, rx1, ry1 = r
        crop = rendered_img.crop((rx0, ry0, rx1, ry1))
        fname = f"page_{page_index+1}_region_{region_count}.png"
        outpath = os.path.join(outdir, fname)
        crop.save(outpath, format="PNG")

        caption = match_caption_for_bbox(text_blocks, r, doc[page_index].rect.height, scale_factor=zoom)
        if not caption:
            # fallback OCR on region to find any textual label inside image
            ocr_text = ocr_image_region(crop)
            caption = ocr_text if ocr_text else None

        results.append({
            "page": page_index + 1,
            "path": outpath,
            "width": crop.width,
            "height": crop.height,
            "source": "rendered_region",
            "bbox": r,
            "caption": caption,
        })
        region_count += 1

    # 6) Sort results by vertical position (top to bottom) so display order is natural
    def vpos(item):
        bbox = item.get("bbox")
        if bbox:
            return bbox[1]
        # if no bbox, return large number to push at end
        return 1e9

    results_sorted = sorted(results, key=vpos)
    return results_sorted


# --------------------------- Full-document processing ---------------------------

def process_pdf(pdf_path_or_url: str, outdir: str, pages: list = None, render_zoom: int = 3):
    ensure_dir(outdir)
    is_url = pdf_path_or_url.lower().startswith("http")
    tmp_pdf = os.path.join(outdir, "_downloaded.pdf") if is_url else pdf_path_or_url

    if is_url:
        print(f"Downloading PDF from {pdf_path_or_url} ...")
        r = requests.get(pdf_path_or_url)
        r.raise_for_status()
        with open(tmp_pdf, "wb") as f:
            f.write(r.content)

    # open doc with PyMuPDF to know page count
    doc = fitz.open(tmp_pdf)
    page_count = doc.page_count
    doc.close()

    if pages is None:
        pages_to_process = list(range(page_count))
    else:
        pages_to_process = [p for p in pages if 0 <= p < page_count]

    out = {"source": pdf_path_or_url, "pages": {}}
    for p in pages_to_process:
        print(f"Processing page {p+1}/{page_count} ...")
        page_outdir = os.path.join(outdir, f"page_{p+1}")
        ensure_dir(page_outdir)
        page_results = process_page(tmp_pdf, p, page_outdir, render_zoom=render_zoom)
        out["pages"][p+1] = page_results

    # write JSON summary
    summary_path = os.path.join(outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Done. Summary written to {summary_path}")
    return out


# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF hybrid image/diagram extractor prototype")
    parser.add_argument("--url", type=str, help="PDF URL to download and process", default=None)
    parser.add_argument("--pdf", type=str, help="Local PDF path to process", default=None)
    parser.add_argument("--outdir", type=str, help="Output directory", default="./pdf_output")
    parser.add_argument("--pages", type=str, help="Comma-separated page indices (1-based) to process", default=None)
    parser.add_argument("--zoom", type=int, help="Render zoom factor (integer, e.g., 2-6)", default=3)
    args = parser.parse_args()

    if not args.url and not args.pdf:
        parser.error("Provide --url or --pdf")

    src = args.url if args.url else args.pdf
    if args.pages:
        pages = [int(x) - 1 for x in args.pages.split(",")]
    else:
        pages = None

    process_pdf(src, args.outdir, pages=pages, render_zoom=args.zoom)