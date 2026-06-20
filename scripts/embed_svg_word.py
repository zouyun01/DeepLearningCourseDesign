#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-process the report: replace every embedded PNG with its vector SVG.

python-docx cannot embed SVG, so the docx is first built with PNGs. This script
then uses Word (COM automation) to swap each PNG for the matching SVG, which Word
stores as a true vector graphic (with a PNG fallback) -> lossless, scalable, and
the carefully-built two-column / spanning layout is preserved (each picture is
replaced in place, keeping its paragraph, section break and display size).

Matching is by byte hash (embedded image bytes == results/figures/<name>.png),
so it is independent of insertion order. Requires: Windows + Word + pywin32.
Word must be CLOSED for the target file.

Run: E:/anaconda3/envs/dl/python.exe scripts/embed_svg_word.py
"""
import glob
import hashlib
import os
import re
import sys
import zipfile

DOCX = os.path.abspath("docs/深度学习大作业_邹研泽_修订.docx")
FIGDIR = "results/figures"


def png_hash_to_svg():
    m = {}
    for png in glob.glob(os.path.join(FIGDIR, "*.png")):
        svg = png[:-4] + ".svg"
        if os.path.exists(svg):
            with open(png, "rb") as f:
                m[hashlib.md5(f.read()).hexdigest()] = os.path.abspath(svg)
    return m


def ordered_svgs(docx):
    """Return the SVG path for each embedded picture, in document order
    (None if no matching SVG exists)."""
    z = zipfile.ZipFile(docx)
    rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
    rid2tgt = dict(re.findall(r'Id="([^"]+)"[^>]*?Target="([^"]+)"', rels))
    doc = z.read("word/document.xml").decode("utf-8")
    h2svg = png_hash_to_svg()
    out = []
    for rid in re.findall(r'r:embed="([^"]+)"', doc):
        tgt = rid2tgt.get(rid, "")
        media = tgt if tgt.startswith("word/") else "word/" + tgt.lstrip("/")
        try:
            data = z.read(media)
        except KeyError:
            out.append(None); continue
        out.append(h2svg.get(hashlib.md5(data).hexdigest()))
    return out


def main():
    svgs = ordered_svgs(DOCX)
    print(f"pictures in document: {len(svgs)}; matched to svg: {sum(x is not None for x in svgs)}")

    import win32com.client as win32
    app = win32.gencache.EnsureDispatch("Word.Application")
    app.Visible = False
    app.DisplayAlerts = 0
    replaced = 0
    try:
        doc = app.Documents.Open(DOCX)
        n = doc.InlineShapes.Count
        if n != len(svgs):
            print(f"[WARN] InlineShapes={n} but xml pictures={len(svgs)}; aborting to be safe")
            doc.Close(False); return
        for i in range(1, n + 1):
            svg = svgs[i - 1]
            if not svg:
                continue
            shp = doc.InlineShapes(i)
            w, h = shp.Width, shp.Height
            rng = shp.Range
            rng.Collapse(0)  # end of the old shape
            pic = doc.InlineShapes.AddPicture(FileName=svg, LinkToFile=False,
                                              SaveWithDocument=True, Range=rng)
            pic.Width, pic.Height = w, h
            doc.InlineShapes(i).Delete()  # remove the old PNG (still at index i)
            replaced += 1
        doc.Save()
        doc.Close(False)
    finally:
        app.Quit()
    print(f"replaced {replaced} PNG -> SVG (vector). Saved {DOCX}")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("requires Windows + Word + pywin32")
    main()
