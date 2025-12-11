"""
extract_sldprt_metadata.py

Script pentru extragerea metadatelor din fișiere SolidWorks .sldprt
folosind SolidWorks API prin pywin32.

Cerințe:
- Windows
- SolidWorks instalat și înregistrat ca aplicație COM
- pywin32 instalat:  pip install pywin32

Utilizare:
    python extract_sldprt_metadata.py --input "C:\cale\la\fisier.sldprt"
    python extract_sldprt_metadata.py --input "C:\folder\cu\piese" --output metadata.csv
"""

import os
import argparse
import csv
import sys
import pythoncom
import win32com.client


def init_solidworks(visible=False):
    """Pornește aplicația SolidWorks prin COM și o returnează."""
    pythoncom.CoInitialize()
    try:
        sw_app = win32com.client.Dispatch("SldWorks.Application")
        sw_app.Visible = visible
        return sw_app
    except Exception as e:
        print("Eroare la conectarea la SolidWorks:", e)
        sys.exit(1)


def extract_metadata_from_file(sw_app, file_path):
    """
    Extrage metadate dintr-un fișier .sldprt folosind instanța SolidWorks dată.

    Returnează un dict cu:
    - nume fișier, cale
    - volum, suprafață, masă
    - bounding box (min/max pe X, Y, Z)
    - număr de features, fețe, muchii, corpuri
    """
    file_path = os.path.abspath(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Fișier inexistent: {file_path}")

    # Tip document: 1 = part
    sw_doc_part = 1
    sw_open_silent = 0  # fără ferestre pop-up

    errors = 0
    warnings = 0

    model = sw_app.OpenDoc6(
        file_path,
        sw_doc_part,
        sw_open_silent,
        "",
        errors,
        warnings
    )

    if model is None:
        raise RuntimeError(f"Nu pot deschide fișierul în SolidWorks: {file_path}")

    # Extension pentru proprietăți de masă & geometrie
    model_ext = model.Extension

    # GetMassProperties2:
    # returnează un vector cu:
    # [0-2]: centru de greutate (x,y,z)
    # [3]: volum
    # [4]: suprafață
    # [5]: masă
    # [6-11]: bounding box (minX, maxX, minY, maxY, minZ, maxZ)
    try:
        mass_props = model_ext.GetMassProperties2(1, 1)  # include corpuri ascunse
    except Exception:
        mass_props = None

    volume = area = mass = None
    bbox = [None] * 6

    if mass_props and len(mass_props) >= 12:
        volume = mass_props[3]
        area = mass_props[4]
        mass = mass_props[5]
        bbox = mass_props[6:12]

    # Număr de features
    feature_count = 0
    feature_types = []

    feat = model.FirstFeature()
    while feat:
        feature_count += 1
        try:
            feature_types.append(feat.GetTypeName2())
        except Exception:
            feature_types.append("Unknown")
        feat = feat.GetNextFeature()

    # Corpuri solide
    body_count = 0
    face_count = 0
    edge_count = 0

    try:
        bodies = model.GetBodies2(0, False)  # 0 = solid bodies
    except Exception:
        bodies = None

    if bodies:
        body_count = len(bodies)
        for body in bodies:
            try:
                faces = body.GetFaces()
                edges = body.GetEdges()
            except Exception:
                faces = None
                edges = None

            if faces:
                face_count += len(faces)
            if edges:
                edge_count += len(edges)

    metadata = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "volume": volume,
        "surface_area": area,
        "mass": mass,
        "bbox_min_x": bbox[0],
        "bbox_max_x": bbox[1],
        "bbox_min_y": bbox[2],
        "bbox_max_y": bbox[3],
        "bbox_min_z": bbox[4],
        "bbox_max_z": bbox[5],
        "feature_count": feature_count,
        "body_count": body_count,
        "face_count": face_count,
        "edge_count": edge_count,
        "feature_types": ";".join(feature_types)  # listă concatenată pentru CSV
    }

    # Închidem documentul (fără a închide SolidWorks)
    sw_app.CloseDoc(model.GetTitle())

    return metadata


def collect_sldprt_files(input_path):
    """Returnează o listă de căi .sldprt dintr-un fișier sau director."""
    input_path = os.path.abspath(input_path)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".sldprt"):
            return [input_path]
        else:
            raise ValueError(f"Fișierul nu este .sldprt: {input_path}")

    if os.path.isdir(input_path):
        sld_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(".sldprt"):
                    sld_files.append(os.path.join(root, f))
        return sld_files

    raise FileNotFoundError(f"Calea nu există: {input_path}")


def write_metadata_to_csv(metadata_list, output_csv):
    """Scrie lista de dicționare de metadate într-un fișier CSV."""
    if not metadata_list:
        print("Nu există metadate de salvat (lista goală).")
        return

    fieldnames = list(metadata_list[0].keys())

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_list:
            writer.writerow(row)

    print(f"[OK] Metadate salvate în: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrage metadate din fișiere SolidWorks .sldprt"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Cale către un fișier .sldprt sau un director cu fișiere .sldprt"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="metadata_sldprt.csv",
        help="Numele fișierului CSV de ieșire (implicit: metadata_sldprt.csv)"
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Dacă este setat, deschide SolidWorks în modul vizibil (debug)"
    )

    args = parser.parse_args()

    # Colectăm fișierele .sldprt
    try:
        sld_files = collect_sldprt_files(args.input)
    except Exception as e:
        print("Eroare la colectarea fișierelor:", e)
        sys.exit(1)

    if not sld_files:
        print("Nu au fost găsite fișiere .sldprt în calea dată.")
        sys.exit(0)

    print(f"Găsite {len(sld_files)} fișiere .sldprt.")

    # Inițializăm SolidWorks
    sw_app = init_solidworks(visible=args.visible)

    all_metadata = []
    for idx, file_path in enumerate(sld_files, start=1):
        print(f"[{idx}/{len(sld_files)}] Procesez: {file_path}")
        try:
            meta = extract_metadata_from_file(sw_app, file_path)
            all_metadata.append(meta)
        except Exception as e:
            print(f"  ⚠ Eroare la fișierul {file_path}: {e}")

    # Opcțional: nu închidem SolidWorks ca să nu deranjăm utilizatorul
    # Dacă vrei să îl închizi mereu:
    # sw_app.Quit()

    if all_metadata:
        write_metadata_to_csv(all_metadata, args.output)
    else:
        print("Nu au putut fi extrase metadate din niciun fișier.")


if __name__ == "__main__":
    main()
