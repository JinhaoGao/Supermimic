import sys
import onnxruntime as ort
a, b = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map for p in sys.argv[1:3])
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(f"DIFF {k}: {str(a.get(k))[:90]} | {str(b.get(k))[:90]}")
print("same keys:", sorted(k for k in a if a.get(k) == b.get(k)))
