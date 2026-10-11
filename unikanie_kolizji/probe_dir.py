import hashlib, glob, os

def h(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]

cands = sorted(set(glob.glob('/home/fujitsu/ros2_ws/**/world_generator.py', recursive=True)))
print("=== distinct world_generator.py under /home/fujitsu/ros2_ws ===")
for p in cands:
    print(h(p), p)

print()
print("=== all pkg dirs named unikanie_kolizji (regex on realpath), with their world_generator hash ===")
seen = set()
for p in cands:
    rp = os.path.realpath(p)
    tag = os.path.dirname(os.path.dirname(rp))
    key = (h(p), tag)
    if key in seen:
        continue
    seen.add(key)
    print(h(p), "  src?=", '/src/' in p, "  install?=", '/install/' in p, "  parent=", tag)
