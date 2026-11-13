"""Launch isolated conda PyQGIS with official activation variables."""
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'outputs/mission-omega/tools/qgis4'
e=os.environ.copy();e.update(PATH=os.pathsep.join(map(str,[p/'Library/bin',p,p/'Scripts']))+os.pathsep+e['PATH'],
    PYTHONPATH=os.pathsep.join(map(str,[p/'Library/python',p/'Library/python/plugins'])),
    QGIS_PREFIX_PATH=str(p/'Library'),QT_PLUGIN_PATH=os.pathsep.join(map(str,[p/'Library/lib/qt6/plugins',p/'Library/plugins'])),
    QTWEBENGINEPROCESS_PATH=str(p/'Library/lib/qt6/QtWebEngineProcess.exe'),
    QTWEBENGINE_RESOURCES_PATH=str(p/'Library/share/qt6/resources'),
    QTWEBENGINE_LOCALES_PATH=str(p/'Library/share/qt6/translations/qtwebengine_locales'),
    QT_QPA_PLATFORM='windows',PROJ_DATA=str(p/'Library/share/proj'),GDAL_DATA=str(p/'Library/share/gdal'),
    OMP_NUM_THREADS='2',QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu')
raise SystemExit(subprocess.call([str(p/'python.exe'),'-X','faulthandler','-u',str(ROOT/'scripts/mission_omega_qgis.py'),*sys.argv[1:]],env=e))
