#!/usr/bin/env python3
"""Generate main-article Fig. 3 from the archived data (see notebooks/fig3.ipynb)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import plot_main_figures as m

if __name__ == '__main__':
    result = m.figure3()
    fig = result[0] if isinstance(result, tuple) else result
    m.save(fig, 'fig3_asymptotics')
    print('Wrote Fig. 3')
