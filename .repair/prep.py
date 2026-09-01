from pathlib import Path

p = Path('.repair/repair.py')
s = p.read_text()
old = "r'''    fn nv12_to_rgba\\(&self\\) -> Option<Vec<u8>> \\{.*?\\n    \\}\\n\\n    fn yuv444p_to_rgba'''"
new = "r'''    fn nv12_to_rgba\\(&self\\) -> Option<Vec<u8>> \\{.*?    fn yuv444p_to_rgba'''"
if old not in s:
    raise RuntimeError('nv12 matcher source not found')
p.write_text(s.replace(old, new, 1))
