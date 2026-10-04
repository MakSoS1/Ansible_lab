#!/usr/bin/env python3
# Blender headless builder for MoveVerse V3 high-detail dance coaches.
# Run:
#   blender --background --factory-startup --python build_coaches_blender.py -- \
#       --specs specs.json --bases /path/bases --out /path/out --limit 1
import argparse, json, math, os, sys, traceback
from pathlib import Path
from mathutils import Vector, Matrix
import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args(argv)


def hx(s):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def mat_principled(name, color, metallic=0.0, rough=0.45, transmission=0.0,
                   alpha=1.0, coat=0.0, emission=None, emission_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    rgb = (*color, 1.0)
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = rgb
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = rough
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    for k in ("Coat Weight", "Clearcoat"):
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = coat
    if emission is not None:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = (*emission, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    m.diffuse_color = (*color, alpha)
    if alpha < 0.999:
        try:
            m.surface_render_method = 'DITHERED'
        except Exception:
            m.blend_method = 'BLEND'
    return m


def style_material(prefix, kind, color):
    if kind in ("holographic", "iridescent"):
        return mat_principled(prefix, color, metallic=0.62, rough=0.18, coat=0.55)
    if kind in ("metallic",):
        return mat_principled(prefix, color, metallic=0.78, rough=0.24, coat=0.35)
    if kind in ("leather",):
        return mat_principled(prefix, color, metallic=0.05, rough=0.28, coat=0.35)
    if kind in ("satin",):
        return mat_principled(prefix, color, metallic=0.08, rough=0.22, coat=0.28)
    if kind in ("nylon", "techwear"):
        return mat_principled(prefix, color, metallic=0.14, rough=0.32, coat=0.24)
    if kind in ("denim", "canvas"):
        return mat_principled(prefix, color, metallic=0.0, rough=0.72)
    if kind in ("sport",):
        return mat_principled(prefix, color, metallic=0.0, rough=0.47)
    return mat_principled(prefix, color, metallic=0.0, rough=0.58)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.armatures,
                       bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        pass


def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not meshes or not arms:
        raise RuntimeError(f"FBX import missing mesh/armature: {path}")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    arm = max(arms, key=lambda o: len(o.data.bones))
    print("BODY", body.name, "verts", len(body.data.vertices))
    print("ARM", arm.name, "bones", len(arm.data.bones))
    print("GROUPS", [g.name for g in body.vertex_groups][:80])
    return body, arm


def obj_bounds_world(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def bone_name(arm, candidates):
    names = [b.name for b in arm.data.bones]
    low = {n.lower(): n for n in names}
    for c in candidates:
        c = c.lower()
        if c in low:
            return low[c]
    for c in candidates:
        c = c.lower()
        for n in names:
            if c in n.lower():
                return n
    return names[0] if names else ""


def add_armature_modifier(obj, arm):
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    obj.parent = arm


def clone_region(body, arm, name, keywords, mat, zlo=0.0, zhi=1.0,
                 inflate=0.006, radial_scale=1.0, lower_flare=0.0,
                 thickness=0.0025):
    src = body.data
    mn, mx = obj_bounds_world(body)
    H = mx.z - mn.z
    groups = []
    for g in body.vertex_groups:
        gl = g.name.lower()
        if any(k.lower() in gl for k in keywords):
            groups.append(g.index)
    use_groups = bool(groups)
    sel = set()
    for v in src.vertices:
        zw = (body.matrix_world @ v.co).z
        zn = (zw - mn.z) / max(H, 1e-6)
        if not (zlo <= zn <= zhi):
            continue
        if use_groups:
            w = 0.0
            for ge in v.groups:
                if ge.group in groups:
                    w = max(w, ge.weight)
            if w < 0.035:
                continue
        sel.add(v.index)
    # Fallback coordinate-only region if a rig export named groups unexpectedly.
    if len(sel) < 80:
        sel = set()
        for v in src.vertices:
            zw = (body.matrix_world @ v.co).z
            zn = (zw - mn.z) / max(H, 1e-6)
            if zlo <= zn <= zhi:
                sel.add(v.index)

    old_to_new = {}
    verts = []
    src_index = []
    faces = []
    face_uvs = []
    uvsrc = src.uv_layers.active
    for p in src.polygons:
        ids = list(p.vertices)
        if not ids or not all(i in sel for i in ids):
            continue
        f = []
        for oi in ids:
            if oi not in old_to_new:
                old_to_new[oi] = len(verts)
                c = src.vertices[oi].co.copy()
                # Inflate in local XY around body center; garments remain
                # compatible with the original skin weights.
                z_world = (body.matrix_world @ c).z
                zn = (z_world - mn.z) / max(H, 1e-6)
                scale = radial_scale + lower_flare * max(0.0, (0.70 - zn) / 0.70)
                c.x *= scale
                c.y *= scale
                verts.append(tuple(c))
                src_index.append(oi)
            f.append(old_to_new[oi])
        faces.append(f)
        if uvsrc:
            face_uvs.append([tuple(uvsrc.data[li].uv) for li in p.loop_indices])

    if not faces:
        return None

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if uvsrc and face_uvs:
        uv = mesh.uv_layers.new(name="UVMap")
        for poly, uvs in zip(mesh.polygons, face_uvs):
            for li, co in zip(poly.loop_indices, uvs):
                uv.data[li].uv = co
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = body.matrix_world.copy()

    # Copy all body vertex weights for the kept vertices so clothing follows
    # the exact same animation deformation as the body.
    for sg in body.vertex_groups:
        ng = obj.vertex_groups.new(name=sg.name)
        vals = []
        for ni, oi in enumerate(src_index):
            try:
                w = sg.weight(oi)
            except RuntimeError:
                continue
            if w > 0.0:
                ng.add([ni], w, "REPLACE")
    add_armature_modifier(obj, arm)

    if mat:
        obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    if inflate > 0:
        solid = obj.modifiers.new("GarmentThickness", "SOLIDIFY")
        solid.thickness = max(thickness, inflate * 0.25)
        solid.offset = 1.0
        solid.use_even_offset = True
    bevel = obj.modifiers.new("SoftGarmentEdges", "BEVEL")
    bevel.width = 0.0016
    bevel.segments = 2
    return obj


def add_cube(name, loc, scale, mat, bevel=0.02):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        md = o.modifiers.new("Rounded", "BEVEL")
        md.width = bevel
        md.segments = 3
    if mat:
        o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def add_uvsphere(name, loc, scale, mat, seg=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def add_cylinder(name, p1, p2, radius, mat, vertices=24):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    L = d.length
    mid = (p1 + p2) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=L, location=mid)
    o = bpy.context.object
    o.name = name
    q = d.to_track_quat('Z', 'Y')
    o.rotation_euler = q.to_euler()
    if mat:
        o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def add_torus(name, loc, major, minor, mat, rot=(0,0,0), major_segments=48, minor_segments=10):
    bpy.ops.mesh.primitive_torus_add(major_radius=major, minor_radius=minor,
                                    major_segments=major_segments, minor_segments=minor_segments,
                                    location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    if mat:
        o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = True
    return o


def parent_to_bone(obj, arm, candidates):
    b = bone_name(arm, candidates)
    if not b:
        return
    mw = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = b
    obj.matrix_world = mw


def add_curve(name, points, radius, mat, resolution=2):
    cu = bpy.data.curves.new(name + "_curve", "CURVE")
    cu.dimensions = "3D"
    cu.resolution_u = resolution
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    spl = cu.splines.new("BEZIER")
    spl.bezier_points.add(len(points) - 1)
    for bp, p in zip(spl.bezier_points, points):
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    if mat:
        o.data.materials.append(mat)
    return o


def convert_to_mesh(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    obj.select_set(False)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def make_pin_group(obj, axis="Z", fraction=0.18, top=True):
    if obj.type != "MESH" or not obj.data.vertices:
        return None
    vals = [v.co.z for v in obj.data.vertices]
    mn, mx = min(vals), max(vals)
    cut = mx - (mx-mn)*fraction if top else mn + (mx-mn)*fraction
    vg = obj.vertex_groups.new(name="CLOTH_PIN")
    for v in obj.data.vertices:
        w = 1.0 if (v.co.z >= cut if top else v.co.z <= cut) else 0.0
        if w:
            vg.add([v.index], w, "REPLACE")
    return vg


def clothify(obj, physics, quality=8, mass=0.25, tension=16, bending=0.45):
    if not obj or obj.type != "MESH":
        return
    pin = make_pin_group(obj)
    try:
        md = obj.modifiers.new("MoveVerseCloth", "CLOTH")
        md.settings.quality = quality
        md.settings.mass = mass
        md.settings.tension_stiffness = tension
        md.settings.compression_stiffness = tension
        md.settings.shear_stiffness = max(5.0, tension * 0.55)
        md.settings.bending_stiffness = bending
        md.settings.vertex_group_mass = pin.name if pin else ""
        md.collision_settings.use_self_collision = True
        md.collision_settings.self_friction = 4.0
        physics.append({"object": obj.name, "type": "cloth", "pin_group": "CLOTH_PIN",
                        "mass": mass, "tension": tension, "bending": bending})
    except Exception as e:
        print("WARN cloth", obj.name, e)


def skirt_mesh(name, mn, mx, waist_z, length, r_top_x, r_top_y, flare, mat, panels=40):
    z0 = waist_z
    z1 = waist_z - length
    verts = []
    faces = []
    rings = 9
    for r in range(rings):
        t = r/(rings-1)
        z = z0*(1-t)+z1*t
        sx = r_top_x * (1 + (flare-1)*t)
        sy = r_top_y * (1 + (flare-1)*t)
        for i in range(panels):
            a = 2*math.pi*i/panels
            verts.append((math.cos(a)*sx, math.sin(a)*sy, z))
    for r in range(rings-1):
        for i in range(panels):
            a = r*panels+i
            b = r*panels+(i+1)%panels
            c = (r+1)*panels+i
            d = (r+1)*panels+(i+1)%panels
            faces.append((a,c,b))
            faces.append((b,c,d))
    me=bpy.data.meshes.new(name+"_mesh")
    me.from_pydata(verts,[],faces); me.update()
    o=bpy.data.objects.new(name,me); bpy.context.collection.objects.link(o)
    if mat: o.data.materials.append(mat)
    for p in me.polygons: p.use_smooth=True
    sol=o.modifiers.new("SkirtThickness","SOLIDIFY"); sol.thickness=0.0022
    return o


def add_panel(name, center, width, length, front_y, mat, flare=0.18, segments=12):
    # Vertical garment panel in XZ plane, slight widening toward the hem.
    verts=[]; faces=[]
    for r in range(segments+1):
        t=r/segments
        z=center.z - t*length
        w=width*(1+flare*t)
        y=front_y + 0.015*math.sin(t*math.pi)
        verts.extend([(-w/2+center.x,y,z),(w/2+center.x,y,z)])
    for r in range(segments):
        a=2*r; b=a+1; c=a+2; d=a+3
        faces.extend([(a,c,b),(b,c,d)])
    me=bpy.data.meshes.new(name+"_mesh"); me.from_pydata(verts,[],faces); me.update()
    o=bpy.data.objects.new(name,me); bpy.context.collection.objects.link(o)
    if mat: o.data.materials.append(mat)
    sol=o.modifiers.new("PanelThickness","SOLIDIFY"); sol.thickness=0.0018
    bev=o.modifiers.new("HemBevel","BEVEL"); bev.width=0.0015; bev.segments=2
    return o


def add_hat(kind, head, dims, mat_primary, mat_secondary=None):
    hx_, hy_, hz = head
    if kind == "cowboy":
        brim=add_uvsphere("CowboyHatBrim",(hx_,hy_,hz+0.08),(dims*.78,dims*.58,0.025),mat_primary,40,12)
        crown=add_uvsphere("CowboyHatCrown",(hx_,hy_,hz+0.13),(dims*.42,dims*.38,dims*.34),mat_primary,32,16)
        return [brim,crown]
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=dims*.52, depth=dims*.30,
                                       location=(hx_,hy_,hz+0.08))
    cap=bpy.context.object; cap.name="BaseballCap"; cap.data.materials.append(mat_primary)
    brim=add_uvsphere("CapBrim",(hx_,hy_-dims*.42,hz+0.06),(dims*.48,dims*.34,0.018),mat_primary,32,10)
    return [cap,brim]


def create_hair(spec, arm, mn, mx, mats, physics):
    style=spec["hair"]["style"]
    color=hx(spec["hair"]["color"])
    accent=hx(spec["hair"].get("accent", spec["hair"]["color"]))
    mat=mat_principled(f"Hair_{spec['id']:02d}",color,metallic=0.02,rough=0.38)
    mat2=mat_principled(f"HairAccent_{spec['id']:02d}",accent,metallic=0.05,rough=0.32)
    H=mx.z-mn.z; W=mx.x-mn.x; D=mx.y-mn.y
    hc=Vector(((mn.x+mx.x)/2,(mn.y+mx.y)/2,mn.z+H*0.91))
    hr=max(W*0.135,H*0.052)
    objects=[]
    cap=add_uvsphere("HairCap",hc+(Vector((0,0,H*.012))), (hr*1.0,hr*.93,hr*.82),mat,40,24)
    parent_to_bone(cap,arm,["head"]); objects.append(cap)
    back_y=mx.y + D*.04

    def strand(points,r=.012,m=mat,cloth=False):
        o=add_curve("HairStrand",points,r,m)
        o=convert_to_mesh(o)
        if cloth:
            clothify(o,physics,quality=6,mass=.12,tension=10,bending=.18)
        parent_to_bone(o,arm,["head"])
        objects.append(o)

    if style in ("buzz","pink_buzz"):
        cap.scale.z=.55
    elif style in ("short","black_short","blond_short","silver_short","messy_short","cap_short"):
        for i in range(14):
            a=2*math.pi*i/14
            root=hc+Vector((math.cos(a)*hr*.55, math.sin(a)*hr*.55, hr*.25))
            tip=root+Vector((math.cos(a)*hr*.45, math.sin(a)*hr*.35, hr*(.35+(i%3)*.12)))
            strand([root,tip],hr*.075,mat2 if i%5==0 else mat)
    elif "bun" in style:
        bx=hr*.72
        for s in (-1,1):
            bun=add_uvsphere("HairBun",(hc.x+s*bx,hc.y,hc.z+hr*.45),(hr*.48,hr*.45,hr*.48),mat,32,20)
            parent_to_bone(bun,arm,["head"]); objects.append(bun)
        if "long" in style:
            for s in (-1,1):
                for j in range(8):
                    root=hc+Vector((s*hr*.65,0,hr*.25))
                    end=Vector((root.x+s*hr*.18,back_y,mn.z+H*.53-j*.002))
                    strand([root,(root+end)*.5+Vector((s*.03,0,.03)),end],hr*.055,mat,True)
    elif style in ("big_curls","braided_dreads","dreads"):
        n=34 if style=="big_curls" else 24
        length=H*(.28 if style=="big_curls" else .34)
        for i in range(n):
            a=2*math.pi*i/n
            root=hc+Vector((math.cos(a)*hr*.72, math.sin(a)*hr*.65, hr*.12))
            pts=[root]
            for k in range(1,7):
                t=k/6
                coil=0.035 if style=="big_curls" else 0.012
                pts.append(Vector((root.x+math.sin(t*math.pi*5+i)*coil,
                                   back_y + math.cos(t*math.pi*4+i)*coil,
                                   root.z-length*t)))
            strand(pts,hr*(.052 if style=="big_curls" else .038),mat2 if i%7==0 else mat,True)
    elif style in ("long_ponytail","silver_pony","pink_twin_tail"):
        roots=[hc+Vector((0,D*.18,hr*.2))]
        if style=="pink_twin_tail":
            roots=[hc+Vector((-hr*.62,D*.06,hr*.12)),hc+Vector((hr*.62,D*.06,hr*.12))]
        for root in roots:
            for j in range(16 if len(roots)==1 else 10):
                side=(j-(7.5 if len(roots)==1 else 4.5))*hr*.05
                end=Vector((root.x+side,back_y+hr*.2,mn.z+H*.48))
                mid=(root+end)*.5+Vector((side*.5,D*.15,H*.02))
                strand([root,mid,end],hr*.045,mat2 if j%5==0 else mat,True)
    else: # very_long / long_dark_streaks / long_blonde / long_under_beanie
        n=32
        for i in range(n):
            a=(i/(n-1)-.5)*math.pi*1.25
            root=hc+Vector((math.sin(a)*hr*.8, math.cos(a)*hr*.28, hr*.05))
            z_end=mn.z+H*(.43 if style=="very_long" else .5)
            end=Vector((root.x+math.sin(a)*hr*.35,back_y+hr*.25,z_end))
            mid=(root+end)*.5+Vector((math.sin(a)*.03,D*.08,H*.015))
            strand([root,mid,end],hr*.038,mat2 if ("streak" in style and i%6==0) else mat,True)
    return objects


def build_outfit(spec, body, arm, mn, mx, physics):
    outf=spec["outfit"]; arch=outf["archetype"]; kind=outf["material"]
    p=hx(outf["primary"]); s=hx(outf["secondary"]); a=hx(outf["accent"])
    mp=style_material(f"M_{spec['id']:02d}_P",kind,p)
    ms=style_material(f"M_{spec['id']:02d}_S",kind,s)
    ma=style_material(f"M_{spec['id']:02d}_A",kind,a)
    black=style_material(f"M_{spec['id']:02d}_Black","techwear",(0.035,0.038,0.05))
    white=style_material(f"M_{spec['id']:02d}_White","satin",(0.92,0.94,0.97))
    H=mx.z-mn.z; W=mx.x-mn.x; D=mx.y-mn.y
    cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
    waist=mn.z+H*.56; chest=mn.z+H*.72; shoulder=mn.z+H*.82
    front=mn.y-D*.03
    objs=[]

    torso=["spine","chest","clav","breast"]
    arms=["upperarm","lowerarm","shoulder"]
    legs=["pelvis","upperleg","lowerleg","thigh","shin"]
    feet=["foot","toe"]

    def shell(n,keys,mat,zlo,zhi,scale=1.02,flare=0,thick=.003):
        o=clone_region(body,arm,n,keys,mat,zlo,zhi,radial_scale=scale,lower_flare=flare,thickness=thick)
        if o: objs.append(o)
        return o

    def pocket(x,z,y,mat=ms,sz=(.065,.025,.08)):
        o=add_cube("CargoPocket",(x,y,z),sz,mat,bevel=.012)
        parent_to_bone(o,arm,["upperleg","thigh","pelvis"])
        objs.append(o)

    # Common shoe shells.
    def shoes(mat=ma, boots=False):
        o=shell("Shoes",feet,mat,.0,.17,1.09,0,.006)
        if boots:
            o2=shell("BootUpper",["lowerleg","shin"],mat,.10,.35,1.075,0,.006)
        return o

    if arch=="crop_bomber_cargo":
        shell("CropTop",torso,black,.61,.78,1.015,0,.003)
        shell("Bomber",torso+arms,mp,.62,.86,1.085,.04,.005)
        shell("CargoPants",legs,black,.08,.60,1.09,.10,.006)
        pocket(cx-W*.16,mn.z+H*.40,front-D*.02); pocket(cx+W*.16,mn.z+H*.40,front-D*.02)
        shoes(ms)
    elif arch=="tech_jacket_cargo":
        shell("TechBase",torso,black,.55,.80,1.02,0,.003)
        shell("TechJacket",torso+arms,mp,.53,.87,1.09,.05,.005)
        shell("TechCargo",legs,mp,.08,.59,1.10,.09,.006)
        pocket(cx-W*.17,mn.z+H*.39,front,ms); pocket(cx+W*.17,mn.z+H*.39,front,ms)
        shoes(ms)
    elif arch=="gold_fringe":
        shell("GoldCrop",torso,mp,.64,.79,1.02,0,.003)
        shell("GoldPants",legs,mp,.08,.59,1.05,.04,.004)
        for side in (-1,1):
            for j in range(16):
                x=cx+side*(W*.10+(j%4)*W*.025)
                z=waist-(j//4)*H*.045
                root=Vector((x,front,z))
                end=root+Vector((side*.018,-.01,-H*(.12+.02*(j%3))))
                o=convert_to_mesh(add_curve("GoldFringe",[root,end],.006,ma))
                parent_to_bone(o,arm,["pelvis","upperleg"]); objs.append(o)
        shoes(ma)
    elif arch=="varsity_cargo":
        shell("WhiteTee",torso,white,.55,.79,1.015,0,.003)
        shell("VarsityJacket",torso,mp,.54,.86,1.09,.04,.005)
        shell("VarsitySleeves",arms,white,.49,.86,1.10,.02,.005)
        shell("BlackCargo",legs,ms,.07,.60,1.10,.10,.006)
        pocket(cx-W*.17,mn.z+H*.39,front,mp); pocket(cx+W*.17,mn.z+H*.39,front,mp)
        shoes(mp)
    elif arch=="holo_pop":
        shell("HoloCrop",torso,mp,.63,.78,1.025,0,.003)
        sk=skirt_mesh("HoloSkirt",mn,mx,waist,H*.12,W*.19,D*.30,1.25,ms,42); objs.append(sk)
        parent_to_bone(sk,arm,["pelvis","root"]); clothify(sk,physics,8,.18,18,.35)
        shell("HoloJacket",torso+arms,ma,.58,.86,1.10,.05,.004)
        shoes(ms,True)
    elif arch=="tank_cargo":
        shell("Tank",torso,mp,.55,.80,1.012,0,.003)
        shell("OliveCargo",legs,ms,.07,.59,1.12,.10,.006)
        pocket(cx-W*.18,mn.z+H*.40,front,ms); pocket(cx+W*.18,mn.z+H*.40,front,ms)
        shoes(black)
    elif arch=="yellow_jacket":
        shell("BlackBase",torso,black,.55,.80,1.015,0,.003)
        shell("YellowJacket",torso+arms,mp,.54,.86,1.095,.05,.005)
        shell("BlackCargo",legs,ms,.07,.59,1.11,.09,.006)
        shoes(mp)
    elif arch=="red_glam_panels":
        shell("RedCrop",torso,mp,.64,.80,1.02,0,.003)
        shell("RedShorts",["pelvis","upperleg"],ms,.46,.60,1.045,0,.004)
        for i,ang in enumerate((-0.9,-0.45,0,0.45,0.9)):
            x=cx+math.sin(ang)*W*.16
            o=add_panel("FlowPanel",Vector((x,0,waist)),W*.12,H*.39,front-D*.02,ms,.35,18)
            o.rotation_euler.z=ang*.25; parent_to_bone(o,arm,["pelvis","root"]); clothify(o,physics,10,.14,12,.18); objs.append(o)
        shoes(black,True)
    elif arch=="sport_harness":
        shell("WhiteSportTop",torso,white,.55,.80,1.02,0,.003)
        shell("WhiteBaggyPants",legs,white,.07,.60,1.12,.13,.006)
        for xoff in (-.15,.15):
            o=add_cylinder("Harness", (cx+xoff*W,front-D*.04,waist-H*.02),
                           (cx+xoff*.45*W,front-D*.04,shoulder-H*.02), .012, ms,18)
            parent_to_bone(o,arm,["spine","chest"]); objs.append(o)
        shoes(ms)
    elif arch=="purple_street":
        shell("BlackCrop",torso,black,.64,.79,1.02,0,.003)
        shell("PurpleJacket",torso+arms,ms,.56,.86,1.10,.05,.005)
        shell("PurpleCargo",legs,mp,.07,.60,1.10,.10,.006)
        shoes(ms)
    elif arch=="floral_shirt_shorts":
        shell("OpenShirt",torso+arms,mp,.56,.84,1.07,.03,.003)
        shell("DenimShorts",["pelvis","upperleg"],style_material("Denim","denim",(0.18,.35,.55)),.42,.60,1.055,0,.004)
        colors=[(1,.22,.25),(.15,.75,.45),(1,.63,.18),(.22,.45,1)]
        for i in range(26):
            x=cx+((i%7)-3)*W*.045; z=mn.z+H*(.60+(i//7)*.055)
            y=front-D*.055
            o=add_uvsphere("FloralPatch",(x,y,z),(.015,.006,.015),mat_principled("Floral",colors[i%4],rough=.55),16,8)
            parent_to_bone(o,arm,["spine","chest"]); objs.append(o)
        shoes(ms)
    elif arch=="purple_active":
        shell("ActiveCrop",torso,mp,.64,.79,1.02,0,.003)
        shell("ActiveShorts",["pelvis","upperleg"],mp,.45,.60,1.04,0,.004)
        shell("ShinyJacket",torso+arms,ms,.56,.86,1.10,.05,.005)
        shoes(ms)
    elif arch=="holo_cutout":
        shell("HoloBody",torso,mp,.59,.80,1.025,0,.003)
        shell("HoloWidePants",legs,ms,.06,.60,1.12,.16,.006)
        for z in (waist+H*.03,waist+H*.13):
            add_torus("HoloStrap",(cx,cy,z),W*.16,.006,ma)
        shoes(mp)
    elif arch=="long_tech_coat":
        shell("TechShirt",torso,black,.55,.80,1.02,0,.003)
        shell("TechCoat",torso+arms,mp,.52,.87,1.10,.05,.005)
        shell("BlackCargo",legs,black,.07,.60,1.08,.08,.005)
        for xoff in (-W*.12,W*.12):
            o=add_panel("CoatTail",Vector((cx+xoff,0,waist+H*.03)),W*.13,H*.42,mx.y+D*.02,ms,.18,18)
            parent_to_bone(o,arm,["pelvis","root"]); clothify(o,physics,9,.20,15,.25); objs.append(o)
        shoes(ms)
    elif arch=="idol_skirt":
        shell("IdolCrop",torso,mp,.64,.80,1.025,0,.003)
        sk=skirt_mesh("IdolSkirt",mn,mx,waist,H*.15,W*.19,D*.29,1.55,white,48); objs.append(sk)
        parent_to_bone(sk,arm,["pelvis"]); clothify(sk,physics,9,.13,12,.22)
        # Puffy sleeves.
        for sx in (-1,1):
            o=add_uvsphere("PuffySleeve",(cx+sx*W*.34,cy,shoulder-H*.04),(W*.10,D*.15,H*.055),mp,28,16)
            parent_to_bone(o,arm,["upperarm"]); objs.append(o)
        shoes(ms,True)
    elif arch=="basketball":
        shell("Jersey",torso,mp,.54,.81,1.045,0,.003)
        shell("BasketShorts",["pelvis","upperleg"],mp,.40,.60,1.09,.04,.004)
        shoes(mp)
    elif arch=="camo_cargo":
        shell("CamoCrop",torso,mp,.63,.79,1.025,0,.003)
        shell("CamoCargo",legs,mp,.07,.60,1.11,.10,.006)
        # Color-block camo pockets/panels.
        for sx in (-1,1):
            pocket(cx+sx*W*.17,mn.z+H*.40,front,ms)
            for zf in (.27,.34,.48):
                o=add_cube("CamoPanel",(cx+sx*W*.13,front-D*.035,mn.z+H*zf),(W*.05,.012,H*.025),ma,.008)
                parent_to_bone(o,arm,["upperleg","lowerleg"]); objs.append(o)
        shoes(black)
    elif arch=="silver_jacket":
        shell("BlackBase",torso,black,.55,.80,1.02,0,.003)
        shell("SilverJacket",torso+arms,mp,.54,.86,1.10,.04,.005)
        shell("BlackBaggy",legs,black,.07,.60,1.11,.12,.006)
        shoes(mp)
    elif arch=="goth_strappy":
        shell("GothCrop",torso,mp,.64,.79,1.02,0,.003)
        shell("GothShort",["pelvis","upperleg"],black,.47,.60,1.04,0,.004)
        o=add_panel("AsymSkirt",Vector((cx-W*.06,0,waist)),W*.27,H*.28,front,ms,.25,14)
        parent_to_bone(o,arm,["pelvis"]); clothify(o,physics,9,.16,15,.25); objs.append(o)
        for z in (mn.z+H*.35,mn.z+H*.45):
            add_torus("ThighStrap",(cx,cy,z),W*.20,.008,ms)
        shoes(ms,True)
    elif arch=="mesh_holo_pants":
        meshmat=mat_principled("MeshTop",(0.03,.03,.04),rough=.35,alpha=.64)
        shell("MeshTop",torso,meshmat,.58,.80,1.02,0,.002)
        shell("HoloWidePants",legs,ms,.06,.60,1.14,.17,.006)
        shoes(black)
    elif arch=="cowgirl_denim":
        shell("DenimCrop",torso,mp,.64,.79,1.025,0,.003)
        shell("DenimShorts",["pelvis","upperleg"],mp,.45,.60,1.05,0,.004)
        shoes(white,True)
        for sx in (-1,1):
            for j in range(7):
                root=Vector((cx+sx*(W*.12+j*W*.018),front,waist))
                end=root+Vector((sx*.015,0,-H*.12))
                o=convert_to_mesh(add_curve("CowgirlFringe",[root,end],.004,white))
                parent_to_bone(o,arm,["pelvis"]); objs.append(o)
    elif arch=="acid_techwear":
        shell("AcidBase",torso,black,.55,.80,1.02,0,.003)
        shell("AcidJacket",torso+arms,mp,.54,.86,1.10,.05,.005)
        shell("AcidCargo",legs,mp,.07,.60,1.10,.10,.006)
        for z in (waist-H*.03, waist+H*.08, chest+H*.02):
            o=add_torus("NeonStrap",(cx,cy,z),W*.18,.008,ms); parent_to_bone(o,arm,["spine","pelvis"]); objs.append(o)
        shoes(ms)
    elif arch=="orange_cargo":
        shell("OrangeCrop",torso,black,.64,.79,1.02,0,.003)
        shell("OrangeCargo",legs,ms,.07,.60,1.11,.10,.006)
        pocket(cx-W*.17,mn.z+H*.40,front,ma); pocket(cx+W*.17,mn.z+H*.40,front,ma)
        shoes(ms)
    elif arch=="pastel_jacket":
        shell("WhiteBase",torso,white,.55,.80,1.02,0,.003)
        shell("PastelJacket",torso+arms,mp,.54,.86,1.10,.05,.005)
        shell("WhiteBaggy",legs,white,.07,.60,1.12,.13,.006)
        shoes(ms)
    else:
        shell("Top",torso,mp,.55,.80,1.03,0,.003)
        shell("Bottom",legs,ms,.07,.60,1.08,.08,.005)
        shoes(ma)

    # Common belts/straps from spec extras.
    extras=set(spec.get("extras",[]))
    if any(x in extras for x in ("pink_straps","utility_straps","black_straps","purple_straps",
                                 "cargo_straps","neon_straps","strap_details","thigh_straps")):
        belt=add_torus("UtilityBelt",(cx,cy,waist),W*.18,.012,ma)
        parent_to_bone(belt,arm,["pelvis","root"]); objs.append(belt)

    return objs, (mp,ms,ma)


def add_accessories(spec, arm, mn, mx, mats):
    H=mx.z-mn.z; W=mx.x-mn.x; D=mx.y-mn.y
    cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
    head=Vector((cx,cy,mn.z+H*.91)); hr=max(W*.12,H*.048)
    front=mn.y-D*.04
    objs=[]
    mp,ms,ma=mats
    extras=set(spec.get("extras",[]))
    if "sunglasses" in extras:
        dark=mat_principled("Lens",(0.02,.025,.035),metallic=.25,rough=.12,alpha=.82)
        frame=mat_principled("Frame",(0.05,.05,.06),metallic=.7,rough=.18)
        for sx in (-1,1):
            tor=add_torus("Sunglasses",(cx+sx*hr*.48,front,head.z+hr*.08),hr*.34,hr*.045,frame,
                          rot=(math.pi/2,0,0),major_segments=40,minor_segments=8)
            parent_to_bone(tor,arm,["head"]); objs.append(tor)
            lens=add_uvsphere("Lens",(cx+sx*hr*.48,front+.002,head.z+hr*.08),(hr*.28,.008,hr*.22),dark,24,12)
            parent_to_bone(lens,arm,["head"]); objs.append(lens)
        br=add_cylinder("GlassesBridge",(cx-hr*.12,front,head.z+hr*.08),(cx+hr*.12,front,head.z+hr*.08),hr*.035,frame,12)
        parent_to_bone(br,arm,["head"]); objs.append(br)
    if "baseball_cap" in extras:
        for o in add_hat("cap",head,hr,mp,ms):
            parent_to_bone(o,arm,["head"]); objs.append(o)
    if "cowboy_hat" in extras:
        for o in add_hat("cowboy",head,hr,ma,mp):
            parent_to_bone(o,arm,["head"]); objs.append(o)
    if "headband" in extras:
        o=add_torus("Headband",(cx,cy,head.z+hr*.22),hr*.80,hr*.07,ma,rot=(math.pi/2,0,0))
        parent_to_bone(o,arm,["head"]); objs.append(o)
    if "hoop_earrings" in extras:
        gold=mat_principled("GoldJewelry",(0.85,.55,.10),metallic=.9,rough=.18)
        for sx in (-1,1):
            o=add_torus("Hoop",(cx+sx*hr*.95,cy,head.z-hr*.10),hr*.25,hr*.035,gold,rot=(math.pi/2,0,0),major_segments=32,minor_segments=8)
            parent_to_bone(o,arm,["head"]); objs.append(o)
    if "chains" in extras or "gold_jewelry" in extras:
        gold=mat_principled("ChainMetal",(0.82,.61,.22),metallic=.92,rough=.16)
        z=mn.z+H*.73
        for i in range(2):
            o=add_torus("NeckChain",(cx,cy-D*.12,z-i*H*.018),W*(.11+i*.018),.0045,gold,rot=(math.pi/2,0,0))
            parent_to_bone(o,arm,["chest","spine"]); objs.append(o)
    return objs


def apply_skin(body, spec):
    c=hx(spec["skin"])
    # Stylized but physically plausible skin. Subsurface deliberately subtle
    # for game export; engines may replace it with their own skin shader.
    m=mat_principled(f"Skin_{spec['id']:02d}",c,metallic=0.0,rough=.47,coat=.08)
    bsdf=m.node_tree.nodes.get("Principled BSDF")
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value=.07
    elif "Subsurface" in bsdf.inputs:
        bsdf.inputs["Subsurface"].default_value=.07
    body.data.materials.clear(); body.data.materials.append(m)
    for p in body.data.polygons: p.use_smooth=True


def setup_stage(mn,mx):
    # Remove old cameras/lights but keep character.
    for o in list(bpy.context.scene.objects):
        if o.type in ("LIGHT","CAMERA"):
            bpy.data.objects.remove(o,do_unlink=True)
    H=mx.z-mn.z; center=Vector(((mn.x+mx.x)/2,(mn.y+mx.y)/2,mn.z+H*.52))
    world=bpy.context.scene.world
    world.color=(.045,.045,.065)
    try:
        world.use_nodes=True
        bg=world.node_tree.nodes.get("Background")
        bg.inputs["Color"].default_value=(.055,.06,.085,1)
        bg.inputs["Strength"].default_value=.42
    except Exception: pass

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=8,location=(center.x,center.y,mn.z-.01))
    floor=bpy.context.object; floor.name="PreviewFloor"
    floor.data.materials.append(mat_principled("FloorMat",(.055,.06,.085),metallic=.18,rough=.30))

    def area(name,loc,energy,size,color):
        data=bpy.data.lights.new(name,"AREA"); data.energy=energy; data.shape="DISK"; data.size=size; data.color=color
        o=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(o); o.location=loc
        q=(center-o.location).to_track_quat('-Z','Y'); o.rotation_euler=q.to_euler()
        return o
    area("Key",(center.x-H*.65,center.y-H*.85,mn.z+H*1.12),1100,H*.75,(1.0,.52,.78))
    area("Fill",(center.x+H*.70,center.y-H*.55,mn.z+H*.85),850,H*.65,(.38,.62,1.0))
    area("Rim",(center.x,center.y+H*.72,mn.z+H*.98),1200,H*.55,(.28,.75,1.0))
    return center,H


def look_at(obj, target):
    direction=Vector(target)-obj.location
    obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()


def render_views(outdir, spec, mn, mx):
    center,H=setup_stage(mn,mx)
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE_NEXT" if hasattr(scene,"eevee") or bpy.app.version >= (4,2,0) else "BLENDER_EEVEE"
    scene.render.resolution_x=640; scene.render.resolution_y=900; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.film_transparent=False
    scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.look="AgX - Medium High Contrast" if "AgX - Medium High Contrast" in [i.name for i in bpy.types.ColorManagedViewSettings.bl_rna.properties['look'].enum_items] else scene.view_settings.look
    data=bpy.data.cameras.new("PreviewCamera")
    cam=bpy.data.objects.new("PreviewCamera",data); bpy.context.collection.objects.link(cam)
    scene.camera=cam
    data.lens=58
    data.sensor_width=36

    views=[
        ("front_minus_y",Vector((center.x,center.y-H*1.75,mn.z+H*.55))),
        ("front_plus_y",Vector((center.x,center.y+H*1.75,mn.z+H*.55))),
        ("three_quarter",Vector((center.x+H*.92,center.y-H*1.52,mn.z+H*.58))),
    ]
    paths=[]
    for name,pos in views:
        cam.location=pos; look_at(cam,Vector((center.x,center.y,mn.z+H*.55)))
        p=outdir/f"{spec['id']:02d}_{name}.png"
        scene.render.filepath=str(p)
        bpy.ops.render.render(write_still=True)
        paths.append(str(p))
    # Keep character-only scene clean for GLB after render by deleting stage.
    for o in list(scene.objects):
        if o.name.startswith("Preview") or o.name in ("Key","Fill","Rim"):
            bpy.data.objects.remove(o,do_unlink=True)
    return paths


def count_tris():
    deps=bpy.context.evaluated_depsgraph_get()
    total=0; verts=0
    for o in bpy.context.scene.objects:
        if o.type!="MESH" or o.name=="PreviewFloor": continue
        e=o.evaluated_get(deps)
        me=e.to_mesh(); me.calc_loop_triangles()
        total += len(me.loop_triangles); verts += len(me.vertices)
        e.to_mesh_clear()
    return verts,total


def export_character(outdir,spec,arm,physics):
    cid=spec["id"]; tag=f"coach_{cid:02d}_{spec['name'].lower()}"
    srcdir=outdir/"source"; glbdir=outdir/"glb"
    srcdir.mkdir(parents=True,exist_ok=True); glbdir.mkdir(parents=True,exist_ok=True)
    blend=srcdir/f"{tag}.blend"; glb=glbdir/f"{tag}_lod0.glb"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    # Select character objects only.
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.context.scene.objects:
        if o.type in ("MESH","ARMATURE","EMPTY"):
            o.select_set(True)
    bpy.context.view_layer.objects.active=arm
    try:
        bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB",use_selection=True,
                                  export_animations=True,export_skins=True,
                                  export_morph=True,export_yup=True)
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB",use_selection=True)
    (srcdir/f"{tag}_physics.json").write_text(json.dumps({
        "character":tag,
        "runtime_note":"Blender Cloth settings are retained in .blend. glTF does not standardize cloth simulation; recreate runtime cloth/secondary-motion from these named objects/pin groups.",
        "items":physics,
    },indent=2),encoding="utf-8")
    return blend,glb


def build_character(spec,bases_dir,outdir):
    reset_scene()
    tag=f"coach_{spec['id']:02d}_{spec['name'].lower()}"
    fbx=bases_dir/tag/f"{tag}.fbx"
    if not fbx.exists():
        raise FileNotFoundError(fbx)
    body,arm=import_fbx(fbx)
    apply_skin(body,spec)
    mn,mx=obj_bounds_world(body)
    physics=[]
    outfit,mats=build_outfit(spec,body,arm,mn,mx,physics)
    hair=create_hair(spec,arm,mn,mx,mats,physics)
    acc=add_accessories(spec,arm,mn,mx,mats)
    # Update bounds after clothing/hair for framing.
    allmesh=[o for o in bpy.context.scene.objects if o.type=="MESH"]
    pts=[]
    for o in allmesh:
        pts += [o.matrix_world@Vector(c) for c in o.bound_box]
    mn2=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts)))
    mx2=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
    preview_dir=outdir/"previews"/tag; preview_dir.mkdir(parents=True,exist_ok=True)
    views=render_views(preview_dir,spec,mn2,mx2)
    verts,tris=count_tris()
    blend,glb=export_character(outdir,spec,arm,physics)
    rec={"id":spec["id"],"name":spec["name"],"tag":tag,"vertices_evaluated":verts,
         "triangles_evaluated":tris,"blend":str(blend),"glb":str(glb),
         "previews":views,"physics_objects":len(physics),
         "body_groups":[g.name for g in body.vertex_groups]}
    print("RESULT",json.dumps(rec))
    return rec


def main():
    args=parse_args()
    specs=json.load(open(args.specs,"r",encoding="utf-8"))["characters"]
    if args.limit: specs=specs[:args.limit]
    bases=Path(args.bases); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    results=[]
    for s in specs:
        try:
            results.append(build_character(s,bases,out))
        except Exception:
            traceback.print_exc()
            raise
    (out/"build_report.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
