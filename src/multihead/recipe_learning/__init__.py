"""Recipe learning workflow - learn optimal recipes from BotVibes experts.

This module enables MultiHead to improve its own recipes by:
1. Querying BotVibes experts for recipe design
2. Benchmarking proposed recipes on test data
3. Evaluating via consensus whether to adopt
4. Tracking recipe versions in SolverRegistry
5. Sharing successes back to the knowledge network

The key insight: "BotVibes knows better" - external experts can design
better recipes than manual ones.
"""

from ._learner import RecipeLearner
from ._workflow import learn_recipe_workflow

__all__ = [
    "RecipeLearner",
    "learn_recipe_workflow",
]
