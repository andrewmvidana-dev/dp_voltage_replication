"""Validate the final PDF and create a portable LaTeX/data source archive."""
from pathlib import Path
from hashlib import sha256
import json
import zipfile
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'output'/'pdf'
PDF = OUT/'grid_privacy_paper.pdf'
DATA = ROOT/'results'/'threat_model_20261013'
reader = PdfReader(PDF)
pages = [p.extract_text() or '' for p in reader.pages]
assert len(pages) == 5, f'Expected five pages, got {len(pages)}'
assert all(len(p.split()) > 120 for p in pages), 'Unexpected near-empty page'
all_text = '\n'.join(pages)
assert 'Andrew Vidana' in all_text and 'Mohammed Khajenejad' in all_text
assert all(x in all_text for x in ['51.385', '51.255', '794.275', '580.672', '786.374'])
assert '??' not in all_text, 'Unresolved LaTeX cross-reference'
manifest = {
    'title': str(reader.metadata.title),
    'authors': ['Andrew Vidana', 'Mohammed Khajenejad'],
    'prepared_date': '2026-10-18',
    'page_count': len(pages),
    'page_words': [len(p.split()) for p in pages],
    'format': 'IEEEtran conference, US Letter, two columns, normal class typography',
    'pdf_sha256': sha256(PDF.read_bytes()).hexdigest(),
    'source_data_sha256': {},
    'experiments_rerun_for_manuscript': False,
    'author_details': 'User-provided spelling; affiliations intentionally omitted.',
}
files = ['comparison.json','key_compromise.json','voltage_attack.json','timing.json','distributed_verification.json']
for name in files:
    manifest['source_data_sha256'][name] = sha256((DATA/name).read_bytes()).hexdigest()
(OUT/'paper_validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(OUT/'grid_privacy_paper_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in ['grid_privacy_paper.tex','make_figures.py','README.md']:
        archive.write(HERE/name,name)
    for path in sorted((HERE/'figures').iterdir()):
        if path.suffix in ['.pdf','.svg','.png']:
            archive.write(path,'figures/'+path.name)
    for name in files:
        archive.write(DATA/name,'data/'+name)
    archive.writestr('paper_validation.json',json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
print('Source archive:',OUT/'grid_privacy_paper_source.zip')
