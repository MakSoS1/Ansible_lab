import bpy, json, sys, traceback
from pathlib import Path

ROOT = Path(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else Path('artifacts/moveverse_coach_pack_50/models/lod0')
OUT = ROOT.parents[1] / 'reports' / 'blender_validation.json'
results=[]

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.actions):
        pass

for glb in sorted(ROOT.glob('coach_*_lod0.glb')):
    rec={'file':glb.name,'ok':False}
    try:
        clear_scene()
        before_actions=set(a.name for a in bpy.data.actions)
        bpy.ops.import_scene.gltf(filepath=str(glb))
        meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
        arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
        tris=0
        for o in meshes:
            deps=bpy.context.evaluated_depsgraph_get()
            e=o.evaluated_get(deps)
            me=e.to_mesh()
            me.calc_loop_triangles(); tris += len(me.loop_triangles)
            e.to_mesh_clear()
        new_actions=[a.name for a in bpy.data.actions if a.name not in before_actions]
        bones=sum(len(a.data.bones) for a in arms)
        rec.update({
            'ok': bool(meshes) and bool(arms) and bones>=20,
            'mesh_objects':len(meshes),'armatures':len(arms),'bones':bones,
            'triangles':tris,'actions':sorted(new_actions),
        })
        if not rec['ok']:
            rec['error']='missing expected mesh/armature/bone count'
    except Exception as e:
        rec['error']=f'{type(e).__name__}: {e}'
        rec['traceback']=traceback.format_exc(limit=3)
    results.append(rec)
    print(glb.name, 'OK' if rec['ok'] else 'FAIL', rec.get('triangles'), rec.get('bones'), rec.get('actions'))

summary={
    'files_checked':len(results),
    'files_ok':sum(1 for r in results if r['ok']),
    'all_ok':all(r['ok'] for r in results) and len(results)==50,
    'results':results,
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(summary,indent=2),encoding='utf-8')
if not summary['all_ok']:
    raise SystemExit(2)
