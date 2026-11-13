"""Official PyQGIS/Qgis2threejs terrain export integration, no meshing code."""
from pathlib import Path
import json, os, sys, time
from qgis.core import (QgsApplication,QgsProject,QgsRasterLayer,QgsCoordinateReferenceSystem,
    QgsRectangle,QgsMapSettings,QgsPoint,Qgis)
from qgis.PyQt.QtCore import QSize,Qt,QCoreApplication
from qgis.PyQt.QtGui import QColor

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1]; variant=sys.argv[2] if len(sys.argv)>2 else 'v3-mad'
raw=OUT/'raw'/scene/variant; dest=OUT/'asset'/f'{scene}-{variant}'
dest.mkdir(parents=True,exist_ok=True); (OUT/'gis').mkdir(exist_ok=True)
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
from PyQt6.QtWebEngineWidgets import QWebEngineView
app=QgsApplication([],True);app.initQgis()
project=QgsProject.instance();crs=QgsCoordinateReferenceSystem('EPSG:4547');project.setCrs(crs)
dem=QgsRasterLayer(str(raw/'dtm.tif'),'UAV-only inferred DTM')
photographic=(raw/'orthographic_photo_mosaic.tif').exists()
color=QgsRasterLayer(str(raw/('orthographic_photo_mosaic.tif' if photographic else 'point_color_projection.tif')),
    'UAV photo-textured DTM orthographic mosaic' if photographic else 'UAV point-color projection (NOT orthophoto)')
assert dem.isValid() and color.isValid()
project.addMapLayer(dem);project.addMapLayer(color)
from qgis.core import QgsRasterDemTerrainProvider
provider=QgsRasterDemTerrainProvider();provider.setLayer(dem);provider.setScale(1.)
project.elevationProperties().setTerrainProvider(provider)
qa=json.loads((raw/'raster_qa.json').read_text()); extent=QgsRectangle(*qa['roi_bounds'])
ms=QgsMapSettings();ms.setDestinationCrs(crs);ms.setLayers([color]);ms.setExtent(extent)
ms.setOutputSize(QSize(int(extent.width()*2),int(extent.height()*2)))
ms.setBackgroundColor(QColor(0,0,0,0))
project.viewSettings().setDefaultViewExtent(__import__('qgis.core',fromlist=['QgsReferencedRectangle']).QgsReferencedRectangle(extent,crs))
project.setFilePathStorage(Qgis.FilePathType.Relative)
project.write(str(OUT/'gis'/f'{scene}-{variant}.qgz'))
if '--native' in sys.argv:
    from qgis._3d import Qgs3D,Qgs3DMapCanvas,Qgs3DMapSettings,QgsDemTerrainSettings,Qgs3DMapExportSettings
    Qgs3D.initialize()
    canvas=Qgs3DMapCanvas();canvas.resize(640,480)
    settings=canvas.mapSettings();settings.setCrs(crs);settings.setTransformContext(project.transformContext())
    from qgis.core import QgsVector3D
    settings.setOrigin(QgsVector3D(extent.center().x(),extent.center().y(),0));settings.setExtent(extent)
    settings.setLayers([color]);settings.setTerrainVerticalScale(1.)
    terrain=QgsDemTerrainSettings();terrain.setLayer(dem);terrain.setResolution(256);terrain.setSkirtHeight(0)
    settings.setTerrainSettings(terrain)
    for _ in range(30):app.processEvents();time.sleep(.1)
    export=Qgs3DMapExportSettings();export.setSceneFolderPath(str(dest));export.setSceneName(scene+'_terrain')
    export.setExportTextures(True);export.setTerrainExportEnabled(True);export.setTerrainResolution(384)
    export.setTerrainTextureResolution(2048);export.setScale(1.)
    print('native export',canvas.scene().exportScene(export),flush=True)
else:
    # Upstream plugin is configured through its public API. Mesh mode preserves NoData holes.
    sys.path.insert(0,str(OUT/'tools'))
    import importlib
    pkg=importlib.import_module('Qgis2threejs-master')
    conf=importlib.import_module('Qgis2threejs-master.conf');conf.GRID_DEM_OUTPUT_MODE='mesh'
    importlib.import_module('Qgis2threejs-master.utils.logging').configureLoggers(log_to_stream=True)
    ExportSettings=importlib.import_module('Qgis2threejs-master.core.exportsettings').ExportSettings
    ModelExporter=importlib.import_module('Qgis2threejs-master.core.export.export').ModelExporter
    const=importlib.import_module('Qgis2threejs-master.core.const')
    settings=ExportSettings();settings.setMapSettings(ms);settings.updateLayers()
    settings.set(ExportSettings.SCENE,dict(checkBox_FixAspectRatio=False,radioButton_FixedExtent=True,
        lineEdit_CenterX=str(extent.center().x()),lineEdit_CenterY=str(extent.center().y()),
        lineEdit_Width=str(extent.width()),lineEdit_Height=str(extent.height()),lineEdit_zFactor='1',comboBox_xyShift=True))
    for layer in settings.layers():
        layer.visible=layer.layerId==dem.id()
        if layer.visible:
            layer.properties.update(radioButton_Tiles=True,checkBox_Sides=False,checkBox_Frame=False,
                comboBox_Resampling='1',materials=[dict(id='uavcolor',type=const.DEMMtlType.LAYER,
                properties=dict(layerIds=[color.id()],comboBox_TextureSize=2048,radioButton_PNG=True,
                    checkBox_TransparentBackground=True,checkBox_Shading=True))])
    settings.saveSettings(str(dest/'export.qto3settings'))
    settings.saveSettings(str(OUT/'gis'/f'{scene}-{variant}.qto3settings'))
    if '--web' in sys.argv:
        ThreeJSExporter=importlib.import_module('Qgis2threejs-master.core.export.export').ThreeJSExporter
        exporter=ThreeJSExporter(settings=settings);exporter.setMapSettings(ms)
        exporter.export(str(dest/'index.html'))
        print('web export files',[(p.name,p.stat().st_size) for p in dest.iterdir()],flush=True)
        os._exit(0)
    # Upstream master imports the webview package rather than webview.webview.
    # Bind the existing public factory functions; no upstream algorithm is changed.
    webpkg=importlib.import_module('Qgis2threejs-master.gui.webview')
    webimpl=importlib.import_module('Qgis2threejs-master.gui.webview.webview')
    # Use upstream in-process exporter classes in a headless process.
    webpkg.getWebViewClass=lambda t,m:webimpl.getWebViewClass(t,0)(None)
    webpkg.getWebPageClass=lambda t,m:webimpl.getWebPageClass(t,0)
    print('Creating upstream ModelExporter',flush=True)
    exporter=ModelExporter(settings);print('Exporter created',flush=True)
    exporter.setMapSettings(ms);exporter.initWebPage(256,256)
    result=exporter.export(str(dest/f'{scene}_terrain.gltf'))
    print('plugin export result',result,flush=True)
    for _ in range(30):app.processEvents();time.sleep(.1)
print('files',[(p.name,p.stat().st_size) for p in dest.iterdir()],flush=True)
# Qt3D destruction ordering can crash on exit; all persistent writes above have completed.
os._exit(0)
