"""Long-run exploration of bounded parameter spaces (novelty + local iteration)."""

from song_analyzer.explore.history import RunRecord, append_record, load_history
from song_analyzer.explore.param_space import ParamSpace, preset_dense_eval, preset_nsynth_tune
from song_analyzer.explore.proposer import Proposal, propose_next
from song_analyzer.explore.runner import exploration_step, run_exploration_loop

__all__ = [
    "ParamSpace",
    "preset_dense_eval",
    "preset_nsynth_tune",
    "RunRecord",
    "append_record",
    "load_history",
    "Proposal",
    "propose_next",
    "exploration_step",
    "run_exploration_loop",
]
