#!/usr/bin/env python3
"""Read-only static audit for a user-supplied Tapped Out APK."""
import argparse, hashlib, json, re, struct, zipfile
from pathlib import Path
HASHES = {'8137faf7a4349043e6e0b0516249c673eb402cf96012c87d805c969cdccae3c2','5b6bf2e4ee386a825e58c3e12aae923d66db8b6fbdd21df05b81b4e1edeb44ae','9eae61f7ae7ae3a8880809e7ffab33e5ae63582d6ad2d941d539d65639a6a47a','7f4d355773c8a7328c9ee405f11b8795ca95fd2dbf7755fa686ea900593cbee3'}
HOSTS = (b'prod.simpsons-ea.com', b'syn-dir.sn.eamobile.com', b'tstodlc.eamobile.com')
def alignment(data):
    if len(data)<64 or data[:4]!=b'\x7fELF' or data[4] not in (1,2) or data[5] not in (1,2): return None
    endian='<' if data[5]==1 else '>'
    try:
        if data[4]==2: off=struct.unpack_from(endian+'Q',data,32)[0]; sz,n=struct.unpack_from(endian+'HH',data,54); minsz,ofs,fmt=56,48,'Q'
        else: off=struct.unpack_from(endian+'I',data,28)[0]; sz,n=struct.unpack_from(endian+'HH',data,42); minsz,ofs,fmt=32,28,'I'
        if n>8192 or not minsz<=sz<=1024 or off+sz*n>len(data): return None
        return [struct.unpack_from(endian+fmt,data,off+i*sz+ofs)[0] for i in range(n) if struct.unpack_from(endian+'I',data,off+i*sz)[0]==1]
    except struct.error: return None
def audit(path):
    path=Path(path)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    out={'name':path.name,'sha256':digest,'byte_size':path.stat().st_size,'libraries':[],'hosts':[],'warnings':[]}
    with zipfile.ZipFile(path) as z:
        names=z.namelist(); out['manifest_present']='AndroidManifest.xml' in names
        out['dex_count']=sum(bool(re.fullmatch(r'classes(?:\d+)?\.dex',x)) for x in names)
        if not out['manifest_present']: out['warnings'].append('Missing AndroidManifest.xml')
        if not out['dex_count']: out['warnings'].append('No classes*.dex')
        for item in z.infolist():
            if not item.filename.startswith('lib/') or not item.filename.endswith('.so'): continue
            if item.file_size>160*1024*1024: out['warnings'].append('Library too large: '+item.filename); continue
            content=z.read(item); sha=hashlib.sha256(content).hexdigest(); aligns=alignment(content)
            row={'path':item.filename,'sha256':sha,'PT_LOAD_p_align':aligns,'elf_16kb':bool(aligns) and all(x%16384==0 for x in aligns),'patcher_support':sha in HASHES if 'libscorpio' in item.filename else None}
            if item.filename.startswith('lib/arm64-v8a/') and not row['elf_16kb']: out['warnings'].append('16 KiB ELF compatibility unknown: '+item.filename)
            out['libraries'].append(row)
            for host in HOSTS:
                if host in content: out['hosts'].append({'source':item.filename,'host':host.decode(),'hits':content.count(host)})
        if not any(x['path'].startswith('lib/arm64-v8a/libscorpio') for x in out['libraries']): out['warnings'].append('No ARM64 Scorpio engine')
        if any(x['patcher_support'] is False for x in out['libraries']): out['warnings'].append('Unknown Scorpio engine hash: manual REA/Ghidra analysis needed')
    out['notice']='Static scan only: no APK signature verification, 16-KiB ZIP packing check, device testing or permission to redistribute EA assets.'
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('apk');p.add_argument('--output');a=p.parse_args()
    result=json.dumps(audit(a.apk),ensure_ascii=False,indent=2)+'\n'
    if a.output: Path(a.output).write_text(result,encoding='utf-8')
    print(result)
