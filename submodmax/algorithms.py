from .deterministic_double_greedy_search import DeterministicDoubleGreedySearch
from .deterministic_local_search import DeterministicLocalSearch
from .deterministic_local_search_pyids import DeterministicLocalSearchPyIDS
from .randomized_double_greedy_search import RandomizedDoubleGreedySearch
from .smooth_local_search import SmoothLocalSearch
from .smooth_local_search_pyids import SmoothLocalSearchPyIDS

algorithms_by_abbreviation = dict(
    SLS=SmoothLocalSearchPyIDS,
    DLS=DeterministicLocalSearchPyIDS,
    DLSRewrite=DeterministicLocalSearch,
    SLSRewrite=SmoothLocalSearch,
    DDGS=DeterministicDoubleGreedySearch,
    RDGS=RandomizedDoubleGreedySearch
)