from determined.cli.cli import main
import cProfile
import pstats
from pstats import SortKey




if __name__ == '__main__':
    cProfile.run('main()', 'clistats')

    p = pstats.Stats('clistats')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()