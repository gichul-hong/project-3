import fitz
import os
import sys

def compress_pdf(input_path, output_path, dpi=100, quality=60):
    if not os.path.exists(input_path):
        print(f"파일을 찾을 수 없습니다: {input_path}", flush=True)
        return

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"원본 파일 크기: {orig_size:.2f} MB", flush=True)

    doc = fitz.open(input_path)
    print(f"총 {len(doc)} 페이지 처리 중...", flush=True)
    new_doc = fitz.open()

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg", jpg_quality=quality)
        
        img_doc = fitz.open("jpeg", img_bytes)
        pdf_bytes = img_doc.convert_to_pdf()
        img_pdf = fitz.open("pdf", pdf_bytes)
        
        new_doc.insert_pdf(img_pdf)
        if (i + 1) % 10 == 0 or (i + 1) == len(doc):
            print(f"  - [{i+1}/{len(doc)}] 페이지 처리 완료", flush=True)

    # garbage=4, deflate=True
    new_doc.save(output_path, garbage=4, deflate=True)
    new_doc.close()
    doc.close()

    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"압축 완료! 결과 파일 크기: {new_size:.2f} MB", flush=True)
    print(f"용량 절감률: {((orig_size - new_size) / orig_size) * 100:.1f}%", flush=True)

if __name__ == "__main__":
    in_pdf = "docs/PDF/Day2.pdf"
    out_pdf = "docs/PDF/Day2_lowres.pdf"
    compress_pdf(in_pdf, out_pdf)
