import json
from pathlib import Path

m = json.loads(Path('data/extracted/manifest.json').read_text(encoding='utf-8'))
print("Total pages:", m['total_pages'])
print("Doc IDs:", set(p['doc_id'] for p in m['pages']))

m['pages'] = m['pages'][:50]
m['total_pages'] = 50

Path('data/extracted/manifest_small.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
print("Done! Created manifest_small.json with 50 pages")