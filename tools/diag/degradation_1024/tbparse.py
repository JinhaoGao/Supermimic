import struct, sys, json, collections
def varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << s; s += 7
        if not x & 0x80: return r, i
def fields(b):
    i = 0
    while i < len(b):
        k, i = varint(b, i); fn, wt = k >> 3, k & 7
        if wt == 0: v, i = varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 5: v = b[i:i+4]; i += 4
        elif wt == 2:
            L, i = varint(b, i); v = b[i:i+L]; i += L
        else: raise ValueError(wt)
        yield fn, wt, v
def read(path):
    out = collections.defaultdict(list)
    data = open(path, 'rb').read(); i = 0
    while i + 12 <= len(data):
        L = struct.unpack('<Q', data[i:i+8])[0]; rec = data[i+12:i+12+L]; i += 12 + L + 4
        step = 0; wall = 0.0; summ = None
        for fn, wt, v in fields(rec):
            if fn == 1 and wt == 1: wall = struct.unpack('<d', v)[0]
            elif fn == 2: step = v
            elif fn == 5: summ = v
        if summ is None: continue
        for fn, wt, v in fields(summ):
            if fn != 1: continue
            tag = None; val = None
            for f2, w2, v2 in fields(v):
                if f2 == 1: tag = v2.decode()
                elif f2 == 2 and w2 == 5: val = struct.unpack('<f', v2)[0]
            if tag is not None and val is not None:
                out[tag].append((step, wall, val))
    return out
if __name__ == '__main__':
    o = read(sys.argv[1])
    json.dump(o, open(sys.argv[2], 'w'))
    for k in sorted(o): print(k, len(o[k]))
