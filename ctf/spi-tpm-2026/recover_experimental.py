#!/usr/bin/env python3
"""Experimental lattice recovery of 128-bit firmware key from 9-bit bilinear leaks.

ONLY OFFLINE SYNTHETIC TESTS until an end-to-end recovery is demonstrated.
The firmware uses a 2^127-1 prime and computes a bilinear mask
L_i = floor((a_i x + b_i y + c_i x*y mod P) / 2^118).
LLL/BKZ for CVP of an augmented 30-dimensional lattice, and enumeration with
the nonlinear xy condition as filter.
"""
import argparse
import json
import os
import math
import random
import time
from pathlib import Path

P = (1 << 127) - 1
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
D1, D2, D3 = (0x62696C696E2D6121, 0x62696C696E2D6221, 0x62696C696E2D6321)
DR = 0x62696C696E2D7274


def sm64(v):
    v = (v + 0x9e3779b97f4a7c15) & MASK64
    v = ((v ^ (v >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    v = ((v ^ (v >> 27)) * 0x94d049bb133111eb) & MASK64
    return v ^ (v >> 31)


def derive(nonce, domain):
    h = ((nonce * 0xd6e8feb86659fd93) & MASK64) ^ domain
    lo = sm64(h)
    hi = sm64(h ^ 0xa5a3564e27f8862f) & ((1 << 63) - 1)
    v = lo | hi << 64
    return v if v else 1


def coeff(nonce):
    return derive(nonce, D1), derive(nonce, D2), derive(nonce, D3)


def public(nonce):
    return sm64(nonce ^ DR) & MASK32


def leakage(nonce, fingerprint):
    value = fingerprint ^ public(nonce)
    if value > 511:
        raise ValueError("fingerprint violates 9-bit bilinear leakage format")
    return value


def synthetic():
    k0 = 0x3dddf9a611e26f3f
    k1 = 0x563598bcddb67486
    gen = random.Random(8945664)
    ob = []
    for i in range(27):
        nonce = gen.getrandbits(32)
        a, b, c = coeff(nonce)
        top9 = (a*k0+b*k1+c*k0*k1) % P >> 118
        ob.append((nonce, public(nonce) ^ top9))
    return ob, (k0, k1)


def build(observations):
    m = len(observations)
    if not 20 <= m <= 27:
        raise ValueError("20-27 independent samples are required")
    from fpylll import IntegerMatrix
    rows = []
    for i in range(m):
        row = [0]*(m+3)
        row[i] = P*1024
        rows.append(row)
    coefficients=[coeff(nonce) for nonce, fp in observations]
    for idx, scalar in [(0,1<<63),(1,1<<63),(2,1)]:
        rows.append([abc[idx]*1024 for abc in coefficients]+
                    [scalar if idx==j else 0 for j in range(3)])
    target=[((2*leakage(n,fp)+1)<<127) for n,fp in observations] + [
        1 << 126, 1 << 126, P//2]
    B = IntegerMatrix.from_matrix(rows)
    return B, target, coefficients


def recover(observations, radii=(2.4,2.6,2.8,3.0,3.2,3.4)):
    from fpylll import LLL, BKZ, GSO, Enumeration, EvaluatorStrategy
    B, target, coefficients = build(observations)
    t0 = time.monotonic()
    LLL.reduction(B, delta=0.99)
    print("LLL time", round(time.monotonic()-t0,2),flush=True)
    BKZ.reduction(B, BKZ.Param(block_size=20, max_loops=2))
    print("BKZ time", round(time.monotonic()-t0,2),flush=True)
    M = GSO.Mat(B, float_type="ld")
    if not M.update_gso(): print("GSO computation unexpectedly did not update")
    m = len(observations)
    k0_col = [int(B[i,m]) for i in range(m+3)]
    k1_col = [int(B[i,m+1]) for i in range(m+3)]
    k2_col = [int(B[i,m+2]) for i in range(m+3)]
    hits=0
    likely=0
    found=[]
    def predicate(coord):
        nonlocal hits,likely
        hits+=1
        xyz=[
            sum(int(round(q))*v for q,v in zip(coord,column))
            for column in (k0_col,k1_col,k2_col)
        ]
        xs,ys,z=xyz
        if xs % (1<<63) or ys % (1<<63):
            return False
        x,y=xs>>63,ys>>63
        if not (0<=x<1<<64 and 0<=y<1<<64 and 0<=z<P):
            return False
        likely+=1
        if z != (x*y)%P: return False
        if all((a*x+b*y+c*z)%P>>118 == leakage(nonce,fp)
               for (nonce,fp),(a,b,c) in zip(observations, coefficients)):
            print("FOUND verified candidate after",hits,
                  "candidates",likely,"secs",round(time.monotonic()-t0,3),flush=True)
            found.append((x,y))
            return True
        return False

    for rad in radii:
        print("Enumerating radius",rad,"elapsed",round(time.monotonic()-t0,1),flush=True)
        enum=Enumeration(M, nr_solutions=1,
                         strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS,
                         callbackf=predicate)
        try:
            enum.enumerate(0,m+3,rad*rad,254,target=M.from_canonical(target))
        except Exception as e:
            print("Enumerate status",type(e).__name__,str(e)[:180],flush=True)
        print("Radius complete",rad,"hits",hits,"bounded",likely,
              "nodes",enum.get_nodes(),"elapsed",round(time.monotonic()-t0,1),flush=True)
        if found:return found[0]
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--synthetic",action="store_true")
    p.add_argument("--json",type=Path)
    p.add_argument("--radius",type=float,default=3.0)
    p.add_argument("--out",type=Path,help="Write recovered key JSON privately; do not log keys")
    a=p.parse_args()
    if a.synthetic == bool(a.json):
        p.error("choose either --synthetic or --json")
    if not (2.2<=a.radius<=3.7):
        p.error("radius outside permitted bounds")
    if a.synthetic:
        ob,expected=synthetic()
        print("SYNTHETIC KEY",tuple(hex(v) for v in expected),flush=True)
    else:
        raw=json.loads(a.json.read_text())
        ob=[(int(x['nonce']),int(x['fingerprint'])) for x in raw['samples']]
        expected=None
    answer=recover(ob, radii=(a.radius,))
    print("ANSWER", "validated" if answer is not None else "no candidate",flush=True)
    if answer is not None and a.out is not None:
        fd=os.open(a.out,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,"w",encoding="utf8") as file:
            json.dump({"key0":str(answer[0]),"key1":str(answer[1])},file)
        print("RECOVERED KEYS SAVED ONLY TO RUNNER TEMPORARY FILE",flush=True)
    if expected is not None:
        assert answer == expected, "FAILED TO RECOVER KNOWN SYNTHETIC SEED"


if __name__=="__main__":
    main()
