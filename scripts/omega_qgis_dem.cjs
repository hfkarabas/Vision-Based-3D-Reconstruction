// Exchange a regular DEM and mesh with Qgis2threejs' upstream GridGeometry.
const fs = require("node:fs");
const path = require("node:path");

const [, , bundle, manifest, output] = process.argv;
const { GridGeometry } = require(path.resolve(bundle));
const grid = JSON.parse(fs.readFileSync(manifest, "utf8"));
const data = fs.readFileSync(path.join(path.dirname(manifest), grid.data));
if (data.byteLength !== grid.columns * grid.rows * 4) {
    throw new Error("Float32 DEM byte count does not match its declared dimensions");
}
const values = new Float32Array(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
);
const geometry = new GridGeometry();
geometry.loadData({
    array: values,
    columns: grid.columns,
    rows: grid.rows,
    extent: { width: grid.width, height: grid.height },
    nodata: new Float32Array([grid.nodata]),
});

// Binary PLY is only the transfer format. Positions and topology are upstream.
const vertices = geometry.getAttribute("position").array;
const indices = geometry.getIndex().array;
const header = [
    "ply", "format binary_little_endian 1.0",
    `element vertex ${vertices.length / 3}`,
    "property float x", "property float y", "property float z",
    `element face ${indices.length / 3}`, "property list uchar uint vertex_indices",
    "end_header", "",
].join("\n");
const faces = Buffer.alloc(indices.length / 3 * 13);
for (let i = 0, j = 0; i < indices.length; i += 3, j += 13) {
    faces[j] = 3;
    faces.writeUInt32LE(indices[i], j + 1);
    faces.writeUInt32LE(indices[i + 1], j + 5);
    faces.writeUInt32LE(indices[i + 2], j + 9);
}
fs.writeFileSync(output, Buffer.concat([
    Buffer.from(header),
    Buffer.from(vertices.buffer, vertices.byteOffset, vertices.byteLength),
    faces,
]), { flag: "wx" });
console.log(JSON.stringify({ vertices: vertices.length / 3, triangles: indices.length / 3 }));
