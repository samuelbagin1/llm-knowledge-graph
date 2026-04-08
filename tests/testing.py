# import os
# from pathlib import Path
# from pdf2image import convert_from_path
# from doclayout_yolo import YOLOv10
# from huggingface_hub import hf_hub_download
# from PIL import Image
#
# # --- Config ---
# PDF_PATH = "./code/assets/ZZ_2004_222_20260101.pdf"
# OUTPUT_DIR = "./code/assets/detected_tables_figures"
# CONF_THRESHOLD = 0.3
# TARGET_CLASSES = {"table", "picture", "figure", "isolate_formula", "formula_caption"}
#
# # --- Setup output dir ---
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # --- Load model ---
# print("Loading DocLayout-YOLO model...")
# model_path = hf_hub_download(
#     repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
#     filename="doclayout_yolo_docstructbench_imgsz1024.pt"
# )
# model = YOLOv10(model_path)
# print("Model loaded.\n")
#
# # --- Convert PDF to images ---
# print(f"Converting PDF to images: {PDF_PATH}")
# pages = convert_from_path(PDF_PATH, dpi=200)
# print(f"Total pages: {len(pages)}\n")
#
# # --- Detect tables and figures ---
# print("Running detection on all pages...\n")
# detections_found = []
#
# for page_num, page_img in enumerate(pages, start=1):
#     results = model.predict(page_img, imgsz=1024, conf=CONF_THRESHOLD, device="cpu")
#
#     for result in results:
#         boxes = result.boxes
#         if boxes is None or len(boxes) == 0:
#             continue
#
#         class_names = result.names  # {0: 'title', 1: 'text', ..., 5: 'table', ...}
#
#         for i, box in enumerate(boxes):
#             cls_id = int(box.cls[0])
#             cls_name = class_names[cls_id].lower()
#             conf = float(box.conf[0])
#
#             if cls_name not in TARGET_CLASSES:
#                 continue
#
#             # Get bounding box coordinates
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
#
#             detections_found.append({
#                 "page": page_num,
#                 "class": cls_name,
#                 "confidence": conf,
#                 "bbox": (x1, y1, x2, y2),
#             })
#
#             # Crop and save the detected region
#             cropped = page_img.crop((x1, y1, x2, y2))
#             filename = f"page{page_num}_{cls_name}_{i}_conf{conf:.2f}.png"
#             cropped.save(os.path.join(OUTPUT_DIR, filename))
#
#             print(f"  Page {page_num:>3d} | {cls_name:<8s} | conf={conf:.2f} | bbox=({x1},{y1},{x2},{y2}) | saved: {filename}")
#
# # --- Summary ---
# print(f"\n{'='*60}")
# print(f"Detection complete.")
# print(f"Total detections (tables + figures + formulas): {len(detections_found)}")
# if detections_found:
#     table_pages = sorted(set(d["page"] for d in detections_found if d["class"] == "table"))
#     figure_pages = sorted(set(d["page"] for d in detections_found if d["class"] in ("picture", "figure")))
#     formula_pages = sorted(set(d["page"] for d in detections_found if d["class"] in ("isolate_formula", "formula_caption")))
#     if table_pages:
#         print(f"Tables found on pages: {table_pages}")
#     if figure_pages:
#         print(f"Figures found on pages: {figure_pages}")
#     if formula_pages:
#         print(f"Formulas found on pages: {formula_pages}")
# print(f"Cropped images saved to: {OUTPUT_DIR}")


# --- New testing code using PDFGraphRAG methods ---
import sys
sys.path.insert(0, "./code")

from pdf_graphrag import PDFGraphRAG

PDF_PATH = "./code/assets/ZZ_2004_222_20260101.pdf"

rag = PDFGraphRAG()

# Step 1: Detect tables, figures, formulas
detections = rag.detect_tables(PDF_PATH)

# Step 2: Group consecutive table detections
groups = PDFGraphRAG.group_table_detections(detections)

print(f"\nFound {len(groups)} table group(s):")
for i, group in enumerate(groups):
    pages = [d["page"] for d in group]
    print(f"  Group {i+1}: pages {min(pages)}-{max(pages)} ({len(group)} detection(s))")

# Step 3: Convert each table group to HTML
for i, group in enumerate(groups):
    image_paths = [d["image_path"] for d in group]
    pages = [d["page"] for d in group]
    print(f"\n{'='*60}")
    print(f"Converting table group {i+1} (pages {min(pages)}-{max(pages)}) to HTML...")
    result = rag.transform_table_to_html(image_paths)
    print(f"Done: {result['html_path']}")
