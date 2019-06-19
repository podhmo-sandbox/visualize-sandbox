from prestring.python import Module
from dictknife import DictWalker
from dictknife import loading

w = DictWalker(["lines"])
d = loading.loadfile(format="json")

r = []
for _, d in w.walk(d):
    if d["language"] == "python" or d["language"] == "py":
        r.append(d["lines"])

m = Module()
m.from_("nbreversible", "code")
for lines in r:
    with m.with_("code()"):
        for line in lines:
            if line.startswith("%"):
                m.stmt("#{}", line)
            else:
                m.stmt(line.rstrip())
    m.sep()
print(m)
