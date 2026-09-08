"""
Stage 1 - Image Plagiarism Pretraining
Flowchart Corpus Parser
--------------------------------------------------------------
Doc corpus dao hinh that (flowchart), tra ve danh sach cap that:
    (source_path, suspicious_path, plag_type, obfuscation, group)

Nhom "shape_based" / "textual_reference": doc source_reference tu
file XML trong thu muc Annotations.
Nhom "hybrid": khong co XML, ghep cap qua hau to so trong ten file
(vd Suspicious_01_00.jpg <-> source_01_00.jpg).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def ensure_images_from_pdfs(root: Path, dpi: int = 150) -> int:
    """
    Nhom 'textual_reference' luu figure duoi dang .pdf (vd
    "suspicious 00.pdf") thay vi anh raster. Ham nay quet de quy 1
    thu muc, render trang dau tien cua moi .pdf CHUA co ban .jpg
    tuong ung (cung ten) thanh anh - lam 1 lan, lan sau chay lai
    se bo qua cac file da convert (kiem tra .jpg da ton tai chua).

    Can thu vien PyMuPDF (pip install pymupdf). Import tre (chi
    import khi thuc su gap file .pdf) de khong bat buoc cai dat
    thu vien nay cho cac corpus khong dung PDF.

    Tra ve so luong file da convert moi trong lan goi nay.
    """
    if not root.exists():
        return 0

    pdf_paths = list(root.rglob("*.pdf"))
    if not pdf_paths:
        return 0

    try:
        import fitz  # PyMuPDF
    except ImportError as error:
        raise ImportError(
            "Thu muc nay co file .pdf can convert sang anh nhung "
            "chua cai PyMuPDF. Chay: pip install pymupdf"
        ) from error

    converted = 0
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for pdf_path in pdf_paths:
        output_path = pdf_path.with_suffix(".jpg")
        if output_path.exists():
            continue

        try:
            document = fitz.open(pdf_path)
            page = document[0]
            pixmap = page.get_pixmap(matrix=matrix)
            pixmap.save(str(output_path))
            document.close()
            converted += 1
        except Exception as error:
            print(f"[Canh bao] Loi convert PDF {pdf_path}: {error}")

    if converted:
        print(f"  Da convert {converted} file PDF -> JPG trong {root}")

    return converted


@dataclass
class PlagiarismPair:
    source_path: Path
    suspicious_path: Path
    plag_type: str
    obfuscation: str
    group: str


def _index_images_by_stem(root: Path) -> dict:
    """
    Quet de quy toan bo anh trong 1 thu muc, tra ve dict
    {ten_file_khong_duoi_VIET_THUONG: duong_dan_day_du}.

    Key duoc lowercase de so khop KHONG PHAN BIET HOA/THUONG, vi
    corpus thuc te thuong dat ten khong nhat quan (vd
    "suspicious-figure-14" trong XML nhung file that ten
    "Suspicious-figure-14.jpg"). Gia tri van la Path goc (dung
    dung ten that tren dia).
    """
    index = {}
    if not root.exists():
        return index

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.stem.lower()] = path

    return index


def _strip_out_suffix(stem: str) -> str:
    """
    XML thuc te khai bao source_reference tro toi file MO TA CAU
    TRUC (vd "Source-figure-325_out.txt", di kem file anh
    "Source-figure-325.jpg" trong cung thu muc Source figures),
    chu khong tro thang toi anh. Can cat "_out" truoc khi tra cuu.
    """
    if stem.endswith("_out"):
        return stem[: -len("_out")]
    return stem


def _extract_attribute(tag_text: str, attribute_name: str) -> str | None:
    match = re.search(
        rf'{attribute_name}\s*=\s*"([^"]*)"',
        tag_text,
    )
    return match.group(1) if match else None


def _fallback_parse_xml_text(xml_text: str) -> dict | None:
    """
    Doc XML bang regex thay vi parser XML chuan.

    Ly do can ham nay: nhieu file XML trong corpus bi loi
    "not well-formed" (thuong do ky tu & chua escape dung chuan
    trong ten file/tieu de), khien xml.etree tu choi doc TOAN BO
    file dung 1 loi nho o 1 cho. Regex khong quan tam file co hop
    le XML hay khong, chi can tim dung 2 the can thiet.
    """
    document_match = re.search(r"<document\b[^>]*>", xml_text)
    if document_match is None:
        return None
    document_reference = _extract_attribute(document_match.group(0), "reference")

    feature_match = re.search(
        r'<feature\s+name\s*=\s*"artificial-plagiarism"[^>]*/?>',
        xml_text,
    )
    if feature_match is None:
        # Khong co nhan dao hinh (tai lieu "sach") -> bo qua, giong
        # logic cua parser chuan.
        return None

    feature_tag = feature_match.group(0)
    source_reference = _extract_attribute(feature_tag, "source_reference")

    if document_reference is None or source_reference is None:
        return None

    return {
        "document_reference": document_reference,
        "plag_type": _extract_attribute(feature_tag, "plag_type") or "unknown",
        "obfuscation": _extract_attribute(feature_tag, "obfuscation") or "unknown",
        "source_reference": source_reference,
    }


def _read_text_any_encoding(path: Path) -> str:
    """
    Thu doc file bang utf-8 truoc, neu loi decode thi fallback
    sang latin-1 (khong bao gio loi decode, chi can du de regex
    tim dung cac gia tri attribute dang la ten file/ASCII).
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _parse_xml_pairs(
    annotations_dir: Path,
    plagiarised_index: dict,
    source_index: dict,
    group_name: str,
) -> list[PlagiarismPair]:
    pairs = []
    recovered_count = 0

    if not annotations_dir.exists():
        print(
            f"[Canh bao] Khong tim thay thu muc annotations: "
            f"{annotations_dir}"
        )
        return pairs

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        print(f"[Canh bao] Khong co file .xml nao trong: {annotations_dir}")

    for xml_path in xml_files:
        parsed = None

        try:
            tree = ElementTree.parse(xml_path)
            root_element = tree.getroot()

            document_reference = root_element.get("reference", "")

            plagiarism_feature = None
            for feature in root_element.findall("feature"):
                if feature.get("name") == "artificial-plagiarism":
                    plagiarism_feature = feature
                    break

            if plagiarism_feature is not None:
                parsed = {
                    "document_reference": document_reference,
                    "plag_type": plagiarism_feature.get("plag_type", "unknown"),
                    "obfuscation": plagiarism_feature.get("obfuscation", "unknown"),
                    "source_reference": plagiarism_feature.get(
                        "source_reference", ""
                    ),
                }
            # plagiarism_feature is None -> tai lieu "sach", bo qua
            # (parsed van la None, khong can fallback).

        except ElementTree.ParseError:
            # XML khong hop le (thuong do ky tu & chua escape) ->
            # thu fallback bang regex thay vi bo qua ca file.
            xml_text = _read_text_any_encoding(xml_path)
            parsed = _fallback_parse_xml_text(xml_text)
            if parsed is not None:
                recovered_count += 1
            else:
                print(
                    f"[Canh bao] Loi doc XML {xml_path} va khong the "
                    f"khoi phuc bang regex - bo qua file nay."
                )

        if parsed is None:
            continue

        suspicious_stem = _strip_out_suffix(
            Path(parsed["document_reference"]).stem.lower()
        )
        source_stem = _strip_out_suffix(
            Path(parsed["source_reference"]).stem.lower()
        )

        suspicious_path = plagiarised_index.get(suspicious_stem)
        source_path = source_index.get(source_stem)

        if suspicious_path is None:
            print(
                f"[Canh bao] Khong tim thay anh suspicious cho "
                f"'{suspicious_stem}' (khai bao trong {xml_path.name})"
            )
            continue

        if source_path is None:
            print(
                f"[Canh bao] Khong tim thay anh source cho "
                f"'{source_stem}' (khai bao trong {xml_path.name})"
            )
            continue

        pairs.append(
            PlagiarismPair(
                source_path=source_path,
                suspicious_path=suspicious_path,
                plag_type=parsed["plag_type"],
                obfuscation=parsed["obfuscation"],
                group=group_name,
            )
        )

    if recovered_count:
        print(
            f"  (khoi phuc duoc {recovered_count} file XML bi loi "
            f"'not well-formed' bang regex fallback)"
        )

    return pairs


_COMPOUND_SUFFIX_PATTERN = re.compile(r"(\d+[_\-]\d+)$")
_SIMPLE_SUFFIX_PATTERN = re.compile(r"(\d+)$")


def _extract_numeric_suffix(stem: str) -> str | None:
    """
    Thu pattern kep truoc (vd "Suspicious_01_00" -> "01_00", dung
    cho nhom Hybrid), neu khong khop thu pattern don (vd
    "suspicious 00" -> "00", dung cho nhom textual_reference).
    """
    match = _COMPOUND_SUFFIX_PATTERN.search(stem)
    if match is not None:
        return match.group(1)

    match = _SIMPLE_SUFFIX_PATTERN.search(stem)
    if match is not None:
        return match.group(1)

    return None


def _parse_hybrid_pairs(
    plagiarised_index: dict,
    source_index: dict,
    group_name: str,
) -> list[PlagiarismPair]:
    """
    Nhom 'hybrid' khong co XML - ghep cap qua hau to so trong ten
    file (vd Suspicious_01_00 <-> source_01_00). plagiarised_index/
    source_index da lowercase key nen khong can xu ly hoa/thuong o day.
    """
    source_by_suffix = {}
    for stem, path in source_index.items():
        suffix = _extract_numeric_suffix(stem)
        if suffix is not None:
            source_by_suffix[suffix] = path

    pairs = []
    unmatched = []

    for stem, suspicious_path in plagiarised_index.items():
        suffix = _extract_numeric_suffix(stem)
        if suffix is None:
            unmatched.append(suspicious_path.name)
            continue

        source_path = source_by_suffix.get(suffix)
        if source_path is None:
            unmatched.append(suspicious_path.name)
            continue

        pairs.append(
            PlagiarismPair(
                source_path=source_path,
                suspicious_path=suspicious_path,
                plag_type=group_name,
                obfuscation="unknown",
                group=group_name,
            )
        )

    if unmatched:
        print(
            f"[Canh bao] {len(unmatched)} anh suspicious trong nhom "
            f"'{group_name}' khong ghep duoc cap (vi du: "
            f"{unmatched[:5]})"
        )

    return pairs


def load_flowchart_pairs(
    flowchart_root: Path,
    groups: list[dict],
) -> list[PlagiarismPair]:
    """
    Doc toan bo corpus dao hinh that, tra ve danh sach cap
    (source, suspicious) tu tat ca cac nhom trong `groups`.
    """
    flowchart_root = Path(flowchart_root)

    if not flowchart_root.exists():
        raise FileNotFoundError(
            f"Khong tim thay FLOWCHART_ROOT: {flowchart_root}\n"
            f"-> Sua duong dan nay trong config.py."
        )

    all_pairs: list[PlagiarismPair] = []

    for group in groups:
        group_root = flowchart_root / group["root"]

        if not group_root.exists():
            print(
                f"[Canh bao] Bo qua nhom '{group['name']}': khong "
                f"tim thay thu muc {group_root}"
            )
            continue

        # Chuyen PDF -> JPG neu co (nhom "textual_reference" dung
        # PDF thay vi anh raster). Vo hai/khong lam gi voi nhom
        # khac (khong co file .pdf nao).
        ensure_images_from_pdfs(group_root / group["plagiarised"])
        ensure_images_from_pdfs(group_root / group["source"])

        plagiarised_index = _index_images_by_stem(
            group_root / group["plagiarised"]
        )
        source_index = _index_images_by_stem(
            group_root / group["source"]
        )

        if not plagiarised_index:
            print(
                f"[Canh bao] Nhom '{group['name']}': khong tim thay "
                f"anh nao trong {group_root / group['plagiarised']}"
            )
        if not source_index:
            print(
                f"[Canh bao] Nhom '{group['name']}': khong tim thay "
                f"anh nao trong {group_root / group['source']}"
            )

        if group["has_xml"]:
            pairs = _parse_xml_pairs(
                annotations_dir=group_root / group["annotations"],
                plagiarised_index=plagiarised_index,
                source_index=source_index,
                group_name=group["name"],
            )
        else:
            pairs = _parse_hybrid_pairs(
                plagiarised_index=plagiarised_index,
                source_index=source_index,
                group_name=group["name"],
            )

        print(f"Nhom '{group['name']}': tim duoc {len(pairs)} cap that.")
        all_pairs.extend(pairs)

    if not all_pairs:
        raise ValueError(
            "Khong parse duoc cap dao hinh that nao. Kiem tra lai "
            "FLOWCHART_ROOT va FLOWCHART_GROUPS trong config.py co "
            "khop voi cau truoc thu muc thuc te khong."
        )

    return all_pairs
