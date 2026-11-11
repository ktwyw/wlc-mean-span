"""Write complete figure exports atomically and verify their bytes.

This helper prevents downstream jobs from seeing an incomplete PDF or PNG.
It changes the file-writing procedure, not the plotted data or calculations.
"""
from __future__ import annotations
from io import BytesIO
import hashlib
import os
from pathlib import Path
import tempfile


def save_figure(fig, path: Path, **kwargs) -> None:
    path = Path(path)
    kind = path.suffix.lower().lstrip('.')
    if kind not in ('pdf', 'png'):
        raise ValueError(f'Unsupported figure format: {kind}')
    if kind == 'pdf':
        # no creation date, so that regenerated PDFs are byte-identical
        kwargs.setdefault('metadata', {'CreationDate': None})
    buffer = BytesIO()
    fig.savefig(buffer, format=kind, **kwargs)
    data = buffer.getvalue()
    if kind == 'pdf' and not (data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')):
        raise IOError(f'Incomplete PDF generated for {path}')
    if kind == 'png' and not (data.startswith(b'\x89PNG\r\n\x1a\n') and data.endswith(b'IEND\xaeB`\x82')):
        raise IOError(f'Incomplete PNG generated for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
            tmp = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(data).digest():
            raise IOError(f'Figure export did not survive write/read verification: {path}')
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
