# MoveVerse 3D Coach Pack Builder

Deterministic builder for 50 stylized dance-coach characters. It emits glTF 2.0 `.glb` files with a shared humanoid rig, embedded demo animations, PBR materials, and four mesh densities (`source`, `lod0`, `lod1`, `lod2`).

The GitHub Actions workflow builds the complete pack, then launches Blender headlessly and imports **all 50 LOD0 models** to verify that the meshes and armatures survive a real Blender/glTF round trip. QA artifacts include one before/after image per coach and a CSV with triangle counts, file sizes, and silhouette preservation.

## Why direct Blender Python instead of Blender MCP on Actions

Blender MCP is designed around an MCP client talking to a live Blender add-on/server. A GitHub Actions runner is non-interactive, so invoking Blender's native `bpy` API in `--background` mode is simpler and more deterministic for CI. The generated assets are still validated by Blender itself; MCP can be added later for interactive art-direction passes on a workstation.

## Output

- `models/source`: dense authoring baseline
- `models/lod0`: gameplay models
- `models/lod1`: medium-distance models
- `models/lod2`: distant/menu models
- `comparisons`: 50 before/after QA images
- `previews`: contact sheet and individual previews
- `reports/optimization_report.csv`: per-model optimization metrics
- `reports/blender_validation.json`: Blender import/armature validation
- `moveverse_coach_pack_50.zip`: complete distributable archive
