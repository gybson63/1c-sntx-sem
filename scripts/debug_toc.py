from sntx_sem.hbk.container import HbkReader, inflate_pack_block
from sntx_sem.hbk.toc_parser import _tokenize

text = (
    inflate_pack_block(HbkReader.from_path("hbk/shquery_root.hbk").get_entity("PackBlock"))
    .decode("utf-8")
    .lstrip("\ufeff")
)
tokens = _tokenize(text)


def parse_block(tokens, pos):
    pos += 1
    items = []
    while tokens[pos] != "}":
        if tokens[pos] == "{":
            sub, pos = parse_block(tokens, pos)
            items.append(sub)
        else:
            t = tokens[pos]
            pos += 1
            if t.lstrip("-").isdigit():
                items.append(int(t))
            elif t.startswith('"'):
                items.append(t.strip('"'))
            else:
                items.append(t)
    pos += 1
    return items, pos


root, _ = parse_block(tokens, 0)
print("root len", len(root), "count", root[0])
print("types", [type(x).__name__ for x in root[:10]])
for i, b in enumerate(root[1:6]):
    print(i, type(b), b if isinstance(b, str) else (b[:6] if isinstance(b, list) else b))
