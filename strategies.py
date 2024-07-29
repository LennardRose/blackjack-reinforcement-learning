import numpy as np

# constants for more readable code
DRAW = 0
STAND = 1
ACTIONS = [STAND, DRAW]


# state = (player_ace, player_sum, dealer_hand[0], self.card_count, player_hand)
def basic_strategy(state):
    player_ace = state[0]
    player_sum = state[1]
    dealer_card = state[2]
    player_hand = state[4]

    assert 12 <= player_sum < 22

    # general standing numbers
    if player_sum >= 19:
        return STAND

    # soft standing numbers
    if player_ace:
        if dealer_card <= 8:
            if player_sum >= 18:
                return STAND
            else:
                return DRAW
        elif dealer_card >= 9:
            return DRAW  # stand should be decided in the first clause

    # hard standing numbers
    else:
        if dealer_card == 2 or dealer_card == 3:
            if player_sum >= 13:
                return STAND
            else:
                return DRAW
        elif 4 <= dealer_card <= 6:
            if player_sum >= 12:
                return STAND
            else:
                return DRAW
        elif 7 <= dealer_card <= 9 or dealer_card == 1:
            if player_sum >= 17:
                return STAND
            else:
                return DRAW
        elif dealer_card == 10:
            if player_sum >= 17:
                return STAND
            elif player_sum == 16:  # "star" special rule table 3.1
                if len(player_hand) == 2:
                    return DRAW
                else:
                    return STAND
            elif player_sum == 14:  # "cross" special rule table 3.1
                if len(player_hand) == 2 and 7 in player_hand:
                    return STAND
                else:
                    return DRAW
            else:
                return DRAW

    raise Exception("No action was taken! state: ", state)

# complete count tables
complete_count_table_hard = np.ones((21, 10))
complete_count_table_hard[17:21, :] = complete_count_table_hard[17:21, :] * -10  # set stand for more
complete_count_table_hard[16, :] = np.array([-0.15, -10, -10, -10, -10, -10, -10, -10, -10, -10])
complete_count_table_hard[15, :] = np.array([0.14, -0.21, -0.25, -0.3, -0.34, -0.35, 0.1, 0.11, 0.06, 0.02])
complete_count_table_hard[14, :] = np.array([0.16, -0.12, -0.17, -0.21, -0.26, -0.28, 0.13, 0.15, 0.12, 0.08])
complete_count_table_hard[13, :] = np.array([10, -0.05, -0.08, -0.13, -0.17, -0.17, 0.20, 0.38, 10, 10])
complete_count_table_hard[12, :] = np.array([10, 0.01, -0.02, -0.05, -0.09, -0.08, 0.50, 10, 10, 10])
complete_count_table_hard[11, :] = np.array([10, 0.14, 0.06, 0.02, -0.01, 0, 10, 10, 10, 10])
complete_count_table_hard[0:11, :] = complete_count_table_hard[0:11, :] * 10  # set draw for less

complete_count_table_soft = np.ones((21, 10))
complete_count_table_soft[0:16, :] = complete_count_table_soft[0:16, :] * 10  # set draw for less
complete_count_table_soft[16:21, :] = complete_count_table_soft[16:21, :] * -10  # set stand for more
complete_count_table_soft[16, 6] = 0.29
complete_count_table_soft[17, 0] = -0.06
complete_count_table_soft[17, 8] = 10
complete_count_table_soft[17, 9] = 0.12


def complete_count_strategy(state):
    player_ace = state[0]
    player_sum = state[1] - 1  # adjust to array
    dealer_card = state[2] - 1  # adjust to array
    card_index = state[3]
    if player_ace:
        return STAND if complete_count_table_soft[player_sum, dealer_card] < card_index else DRAW
    else:
        return STAND if complete_count_table_hard[player_sum, dealer_card] < card_index else DRAW


def mimic_dealer(state):
    if state[1] < 17:
        return DRAW
    else:
        return STAND


def stand_20(state):
    if state[1] < 20:
        return DRAW
    else:
        return STAND


def random_action(state):
    if np.random.randint(2) == STAND:
        return STAND
    else:
        return DRAW

# for sarsa
class epsilon_greedy:
    def __init__(self, e):
        self.e = e


    def epsilon_greedy_strategy(self, state):
        if np.random.randint(0, 1) > self.e:
            return STAND
        else:
            return DRAW


class policy_for_MC():
    """
    for the state values of MC prediction
    """


    def __init__(self, policy_ace, policy_no_ace):
        self.policy_ace = policy_ace
        self.policy_no_ace = policy_no_ace


    def decide_action(self, state):
        player_ace = state[0]
        player_sum = state[1] - 12  # adjust to array
        dealer_card = state[2] - 1  # adjust to array
        card_index = state[3]

        if player_ace:
            return STAND if self.policy_ace[player_sum, dealer_card] > 0 else DRAW
        else:
            return STAND if self.policy_no_ace[player_sum, dealer_card] > 0 else DRAW


assert complete_count_strategy((False, 14, 2, 1, [7, 7])) == STAND  # 1 larger than -0.05
assert complete_count_strategy((True, 17, 7, 1, [7, 7])) == STAND  # 1 larger than -0.05
