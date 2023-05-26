from .base_functions import np2bas, bas2np
from .implication_bases import iter_proper_premises_via_keys, list_pseudo_intents_via_keys
from .mine_equivalence_classes import list_intents_via_LCM, list_keys, list_passkeys
from .order import sort_intents_inclusion, inverse_order
from .indices import linearity_index, distributivity_index

from .orchestrator import explore_data

import pkg_resources
__version__ = pkg_resources.get_distribution('caspailleur').version



