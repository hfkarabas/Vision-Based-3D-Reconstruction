"""Build ready-to-open QGIS projects (labels/buildings/<scene>.qgz) for the labelling kit.

Run with QGIS's Python, e.g.:
  "C:\\Program Files\\QGIS 3.44.15\\bin\\python-qgis-ltr.bat" bench\\make_label_projects.py
Each project: background image, red area outlines (labelled 1..n), and the editable
"buildings" layer (yellow), with the attribute form suppressed and uncertain defaulting to 0.
"""
import json
import sys
from pathlib import Path

from qgis.core import (QgsApplication, QgsDefaultValue, QgsEditFormConfig, QgsFillSymbol, QgsPalLayerSettings,
                       QgsProject, QgsRasterLayer, QgsRectangle, QgsReferencedRectangle, QgsVectorLayer, QgsVectorLayerSimpleLabeling)

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "labels" / "buildings"

app = QgsApplication([], False)
app.initQgis()
for scene in json.loads((KIT / "kit.json").read_text(encoding="utf-8"))["scenes"]:
    name = scene["scene"]
    project = QgsProject.instance()
    project.clear()
    bg = QgsRasterLayer(str(REPO / scene["background"]), "background")
    area = QgsVectorLayer(f"{KIT / (name + '.gpkg')}|layername=area", "area (label inside)", "ogr")
    bld = QgsVectorLayer(f"{KIT / (name + '.gpkg')}|layername=buildings", "buildings", "ogr")
    for lyr in (bg, area, bld):
        if not lyr.isValid():
            sys.exit(f"invalid layer for {name}: {lyr.source()}")
    area.renderer().setSymbol(QgsFillSymbol.createSimple({"style": "no", "outline_color": "255,0,0", "outline_width": "0.8"}))
    pal = QgsPalLayerSettings()
    pal.fieldName = "area_id"
    pal.enabled = True
    area.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    area.setLabelsEnabled(True)
    bld.renderer().setSymbol(QgsFillSymbol.createSimple({"color": "255,230,0,70", "outline_color": "255,230,0",
                                                         "outline_width": "0.6"}))
    idx = bld.fields().indexOf("uncertain")
    bld.setDefaultValueDefinition(idx, QgsDefaultValue("0"))
    form = bld.editFormConfig()
    form.setSuppress(QgsEditFormConfig.SuppressOn)
    bld.setEditFormConfig(form)
    project.setCrs(area.crs())
    for lyr in (bg, area, bld):
        project.addMapLayer(lyr)
    ext = QgsRectangle(area.extent())
    ext.scale(1.15)
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(ext, area.crs()))
    out = KIT / f"{name}.qgz"
    project.write(str(out))
    print("wrote", out)
app.exitQgis()
