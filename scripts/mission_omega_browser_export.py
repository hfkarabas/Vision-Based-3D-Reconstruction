"""Run upstream Qgis2threejs scene and Three.js GLTFExporter in installed Chrome."""
from pathlib import Path
import functools, http.server, json, shutil, socketserver, sys, threading, time
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
scene=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'v3-mad'
dest=OUT/'asset'/f'{scene}-{variant}'
exporter=OUT/'tools/Qgis2threejs-master/node_modules/three/examples/jsm/exporters/GLTFExporter.js'
(dest/'three/exporters').mkdir(exist_ok=True)
shutil.copy2(exporter,dest/'three/exporters/GLTFExporter.js')
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
with socketserver.TCPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(dest))) as server:
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,
            args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1200,'height':800},accept_downloads=True)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('console',lambda msg:print('chrome',msg.type,msg.text[:300]))
        page.goto(f'http://127.0.0.1:{server.server_address[1]}/index.html',wait_until='networkidle')
        page.wait_for_function("async () => {const m=await import('./Qgis2threejs.js');return m.app.scene && Object.values(m.app.scene.mapLayers).some(x=>x.objectGroup.children.length>0)}",timeout=60000)
        page.screenshot(path=str(dest/'browser_scene.png'))
        with page.expect_download(timeout=60000) as download:
            counts=page.evaluate("""async () => {
                const {app}=await import('./Qgis2threejs.js');
                const THREE=await import('three');
                const {GLTFExporter}=await import('./three/exporters/GLTFExporter.js');
                const scene=new THREE.Scene();let meshes=0,vertices=0;
                for(const layer of Object.values(app.scene.mapLayers)) {
                    const group=layer.objectGroup;group.rotation.set(-Math.PI/2,0,0);
                    group.name=layer.properties.name;scene.add(group);
                    group.traverse(o=>{if(o.isMesh){meshes++;vertices+=o.geometry.attributes.position.count;}});
                }
                scene.updateMatrixWorld(true);
                const buffer=await new GLTFExporter().parseAsync(scene,{binary:true});
                const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([buffer],{type:'model/gltf-binary'}));
                a.download='terrain.glb';a.click();return {meshes,vertices,bytes:buffer.byteLength};
            }""")
        download.value.save_as(dest/f'{scene}_terrain.glb')
        (dest/'browser_export.json').write_text(json.dumps(dict(counts=counts,errors=errors,
            exporter='Three.js GLTFExporter, official Qgis2threejs scene; axis conversion follows upstream saveAsGLTF',
            chrome_version=browser.version),indent=2))
        browser.close();print(counts)
    server.shutdown()
