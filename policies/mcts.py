"""Monte Carlo Tree Search policy for the wheelchair grid."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from human_policy_sim.environment import (
    Action,
    GridEnvironment,
    State,
)


@dataclass
class MCTSNode:
    """One node in the Monte Carlo search tree."""

    state: State
    parent: "MCTSNode | None" = None
    action_from_parent: Action | None = None

    children: list["MCTSNode"] = field(
        default_factory=list
    )

    untried_actions: list[Action] = field(
        default_factory=list
    )

    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.total_reward / self.visits


class MCTSPolicy:
    """
    Online Monte Carlo Tree Search policy.

    At every wheelchair state:

        state
          ↓
        MCTS
          ↓
        simulate possible futures
          ↓
        choose best current action

    Unlike DP, no complete policy is computed beforehand.

    Unlike Q-Learning, no Q-table is trained beforehand.
    """

    def __init__(
        self,
        env: GridEnvironment,
        simulations: int = 400,
        exploration_constant: float = math.sqrt(2.0),
        rollout_depth: int = 80,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        self.env = env

        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

        self.rng = random.Random(seed)

        self.actions = (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.STOP,
        )

        self.last_simulations = 0
        self.last_planning_time = 0.0

    def _new_node(
        self,
        state: State,
        parent: MCTSNode | None = None,
        action_from_parent: Action | None = None,
    ) -> MCTSNode:

        return MCTSNode(
            state=state,
            parent=parent,
            action_from_parent=action_from_parent,
            untried_actions=list(self.actions),
        )

    def _uct_score(
        self,
        parent: MCTSNode,
        child: MCTSNode,
    ) -> float:
        """
        Upper Confidence Bound for Trees.

        exploitation + exploration
        """

        if child.visits == 0:
            return float("inf")

        exploitation = child.mean_reward

        exploration = (
            self.exploration_constant
            * math.sqrt(
                math.log(max(parent.visits, 1))
                / child.visits
            )
        )

        return exploitation + exploration

    def _select_child(
        self,
        node: MCTSNode,
    ) -> MCTSNode:

        return max(
            node.children,
            key=lambda child: self._uct_score(
                node,
                child,
            ),
        )

    def _expand(
        self,
        node: MCTSNode,
    ) -> tuple[MCTSNode, float]:

        action_index = self.rng.randrange(
            len(node.untried_actions)
        )

        action = node.untried_actions.pop(
            action_index
        )

        result = self.env.step(
            node.state,
            action,
        )

        child = self._new_node(
            state=result.state,
            parent=node,
            action_from_parent=action,
        )

        node.children.append(child)

        return child, result.reward

    def _rollout_action(
        self,
        state: State,
    ) -> Action:
        """
        Lightweight rollout policy.

        Mostly random, with a small goal-directed preference.

        This is not BFS and does not compute a shortest path.
        """

        # Occasionally choose a purely random action.
        if self.rng.random() < 0.25:
            return self.rng.choice(self.actions)

        goal_row, goal_col = self.env.goal

        row_difference = goal_row - state.row
        col_difference = goal_col - state.col

        desired_heading = None

        # Prefer the dimension with the larger remaining distance.
        if abs(row_difference) >= abs(col_difference):

            if row_difference < 0:
                desired_heading = 0
            elif row_difference > 0:
                desired_heading = 180

        if desired_heading is None:

            if col_difference > 0:
                desired_heading = 90
            elif col_difference < 0:
                desired_heading = 270

        if desired_heading is None:
            return Action.STOP

        current_heading = state.heading.value

        if current_heading == desired_heading:
            return Action.FORWARD

        right_heading = (
            current_heading + 90
        ) % 360

        left_heading = (
            current_heading - 90
        ) % 360

        if right_heading == desired_heading:
            return Action.RIGHT

        if left_heading == desired_heading:
            return Action.LEFT

        # 180-degree difference.
        return self.rng.choice(
            (Action.LEFT, Action.RIGHT)
        )

    def _rollout(
        self,
        state: State,
    ) -> float:
        """Simulate a future trajectory."""

        current_state = state

        total_return = 0.0
        discount = 1.0

        for _ in range(self.rollout_depth):

            if self.env.is_goal(current_state):
                break

            action = self._rollout_action(
                current_state
            )

            result = self.env.step(
                current_state,
                action,
            )

            total_return += (
                discount * result.reward
            )

            if result.terminated:
                break

            current_state = result.state

            discount *= self.gamma

        # Encourage states closer to the target if rollout
        # finishes before reaching the goal.
        if not self.env.is_goal(current_state):

            goal_row, goal_col = self.env.goal

            distance = (
                abs(current_state.row - goal_row)
                + abs(current_state.col - goal_col)
            )

            total_return -= (
                discount * float(distance)
            )

        return total_return

    def _combine_returns(
        self,
        path_rewards: list[float],
        rollout_return: float,
    ) -> float:
        """
        Combine tree-path rewards with a rollout return.

        rollout_return is measured from the rollout start
        state, so each preceding tree reward contributes
        exactly one additional gamma discount.
        """

        total_return = rollout_return

        for reward in reversed(path_rewards):
            total_return = (
                reward
                + self.gamma * total_return
            )

        return total_return

    def _backpropagate(
        self,
        node: MCTSNode,
        path_rewards: list[float],
        rollout_return: float,
    ) -> None:
        """
        Backpropagate discounted action returns.

        A non-root node represents the action taken from its
        parent to reach that node. Therefore its statistics
        must include that transition reward:

            Q(parent, action)
                = reward + gamma * future_return

        This keeps child.mean_reward directly comparable
        during UCT selection and final root-action choice.
        """

        current = node
        future_return = rollout_return
        reward_index = len(path_rewards) - 1

        while current.parent is not None:

            if reward_index < 0:
                raise RuntimeError(
                    "MCTS tree depth and path rewards "
                    "are inconsistent."
                )

            action_return = (
                path_rewards[reward_index]
                + self.gamma * future_return
            )

            current.visits += 1
            current.total_reward += action_return

            future_return = action_return
            reward_index -= 1
            current = current.parent

        # Root has no action_from_parent, but its visit count
        # is required by the UCT exploration term.
        current.visits += 1
        current.total_reward += future_return

        if reward_index != -1:
            raise RuntimeError(
                "Unused MCTS path rewards remained after "
                "backpropagation."
            )

    def action(
        self,
        state: State,
    ) -> Action:
        """Plan from the current state and return one action."""

        if self.env.is_goal(state):
            return Action.STOP

        start_time = time.perf_counter()

        root = self._new_node(state)

        for _ in range(self.simulations):

            node = root

            # Rewards collected while traversing the search tree.
            path_rewards: list[float] = []

            # --------------------------------------------
            # SELECTION
            # --------------------------------------------

            while (
                not node.untried_actions
                and node.children
                and not self.env.is_goal(node.state)
            ):
                child = self._select_child(node)

                transition = self.env.step(
                    node.state,
                    child.action_from_parent,
                )

                path_rewards.append(
                    transition.reward
                )

                node = child

            # --------------------------------------------
            # EXPANSION
            # --------------------------------------------

            if (
                node.untried_actions
                and not self.env.is_goal(node.state)
            ):
                node, expansion_reward = self._expand(
                    node
                )

                path_rewards.append(
                    expansion_reward
                )

            # --------------------------------------------
            # SIMULATION
            # --------------------------------------------

            rollout_return = self._rollout(
                node.state
            )

            # --------------------------------------------
            # COMPLETE DISCOUNTED RETURN
            #
            # Include BOTH:
            #   tree-path rewards
            #   rollout reward
            # --------------------------------------------

            # --------------------------------------------
            # BACKPROPAGATION
            # --------------------------------------------
            #
            # Each tree node stores a return measured from
            # that node's own state. The rollout-start node
            # receives rollout_return directly; ancestors
            # incorporate their transition reward and one
            # gamma discount at each level.
            # --------------------------------------------

            self._backpropagate(
                node,
                path_rewards,
                rollout_return,
            )

        self.last_simulations = self.simulations

        self.last_planning_time = (
            time.perf_counter()
            - start_time
        )

        if not root.children:
            return Action.STOP

        # At decision time choose the action with
        # the best estimated return.
        best_child = max(
            root.children,
            key=lambda child: (
                child.mean_reward,
                child.visits,
            ),
        )

        return best_child.action_from_parent
