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
    {ten_file_khong_duoi: duong_dan_day_du}.
    Neu 2 anh trung ten (khac thu muc con), anh tim thay sau se
    ghi de - nen dat ten file duy nhat trong corpus.
    """
    index = {}
    if not root.exists():
        return index

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.stem] = path

    return index


def _parse_xml_pairs(
    annotations_dir: Path,
    plagiarised_index: dict,
    source_index: dict,
    group_name: str,
) -> list[PlagiarismPair]:
    pairs = []

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
        try:
            tree = ElementTree.parse(xml_path)
        except ElementTree.ParseError as error:
            print(f"[Canh bao] Loi doc XML {xml_path}: {error}")
            continue

        root_element = tree.getroot()

        document_reference = root_element.get("reference", "")
        suspicious_stem = Path(document_reference).stem

        plagiarism_feature = None
        for feature in root_element.findall("feature"):
            if feature.get("name") == "artificial-plagiarism":
                plagiarism_feature = feature
                break

        if plagiarism_feature is None:
            # Khong co nhan dao hinh (co the la doc "sach", khong
            # bi dao) -> bo qua.
            continue

        plag_type = plagiarism_feature.get("plag_type", "unknown")
        obfuscation = plagiarism_feature.get("obfuscation", "unknown")
        source_reference = plagiarism_feature.get("source_reference", "")
        source_stem = Path(source_reference).stem

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
                plag_type=plag_type,
                obfuscation=obfuscation,
                group=group_name,
            )
        )

    return pairs


_SUFFIX_PATTERN = re.compile(r"(\d+[_\-]\d+)$")


def _extract_numeric_suffix(stem: str) -> str | None:
    match = _SUFFIX_PATTERN.search(stem)
    if match is None:
        return None
    return match.group(1)


def _parse_hybrid_pairs(
    plagiarised_index: dict,
    source_index: dict,
    group_name: str,
) -> list[PlagiarismPair]:
    """
    Nhom 'hybrid' khong co XML - ghep cap qua hau to so trong ten
    file (vd Suspicious_01_00 <-> source_01_00).
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
            unmatched.append(stem)
            continue

        source_path = source_by_suffix.get(suffix)
        if source_path is None:
            unmatched.append(stem)
            continue

        pairs.append(
            PlagiarismPair(
                source_path=source_path,
                suspicious_path=suspicious_path,
                plag_type="hybrid",
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
