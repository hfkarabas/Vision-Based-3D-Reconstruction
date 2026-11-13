"""Official Overpass building query; raw response preserved with attribution."""
from pathlib import Path
from datetime import datetime, timezone
import json, urllib.request, urllib.parse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mission-omega'
rows=json.loads((ROOT/'outputs/research-baseline/reports/dataset-audit.json').read_text())['urban']['images']
lat=[float(r['xmp']['GpsLatitude']) for r in rows];lon=[float(r['xmp']['GpsLongitude']) for r in rows]
bbox=[min(lat)-.0005,min(lon)-.0006,max(lat)+.0005,max(lon)+.0006]
query='[out:json][timeout:45];nwr[building]('+','.join(map(str,bbox))+');out geom;'
dest=OUT/'datasets/urban-osm';dest.mkdir(parents=True,exist_ok=True)
(dest/'query.txt').write_text(query)
req=urllib.request.Request('https://overpass-api.de/api/interpreter',data=urllib.parse.urlencode({'data':query}).encode(),
    headers={'User-Agent':'Vision3D-MissionOmega-research/1.0'})
with urllib.request.urlopen(req,timeout=60) as response: data=response.read()
(dest/'response.json').write_bytes(data)
(dest/'provenance.json').write_text(json.dumps(dict(retrieved_at=datetime.now(timezone.utc).isoformat(),
    endpoint=req.full_url,bbox_south_west_north_east=bbox,license='ODbL 1.0',attribution='© OpenStreetMap contributors',
    copyright_url='https://www.openstreetmap.org/copyright',alignment='NOT VERIFIED'),indent=2))
print('elements',len(json.loads(data).get('elements',[])))
