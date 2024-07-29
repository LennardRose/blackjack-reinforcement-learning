from blackjack import *


def MC_prediction(n_episodes, policy, gamma=0, count_cards=False, count_method=None):
    """
    Monte Carlo Prediction, this is the every visit implementation, because for blackjack first and every visit is the same
    :param n_episodes: The number of episodes to calculate the state-values
    :param policy: the policy to evaluate
    :param gamma: the gamma value for the discounts
    :return: the state value estimates, first array for player with an usable ace, second for players without a usable ace
    """
    bj = Blackjack(count_cards=count_cards, count_method=count_method)
    states = (10, 10)  # player sum (12 to 21 possible), dealer card (2-10 + A), no actions

    states_usable_ace = np.zeros(states)
    states_usable_ace_count = np.zeros(states)

    states_no_usable_ace = np.zeros(states)
    states_no_usable_ace_count = np.zeros(states)

    G = 0
    for _ in tqdm(range(n_episodes)):

        episode = bj.play(policy)  # generate episode

        for i, ((ace, player_sum, dealer, _, _), _, reward) in enumerate(reversed(episode)):

            # calculate array index
            row = player_sum - 12
            column = dealer - 1

            G = gamma * G + reward

            if ace:
                states_usable_ace_count[row, column] += 1  # for the average
                states_usable_ace[row, column] += G  # no append, do average by division
            else:
                states_no_usable_ace_count[row, column] += 1  # for the average
                states_no_usable_ace[row, column] += G

    V_ace = np.divide(states_usable_ace, states_usable_ace_count,
                      out=np.zeros_like(states_usable_ace),
                      where=states_usable_ace_count != 0)  # prevent division by zero
    V_no_ace = np.divide(states_no_usable_ace, states_no_usable_ace_count,
                         out=np.zeros_like(states_no_usable_ace),
                         where=states_no_usable_ace_count != 0)

    return None, None, V_ace, V_no_ace # return values not really matching, for plotting in run method necessary


def MC_ES(n_episodes, policy, gamma=0, count_cards=False, count_method=None):
    bj = Blackjack(count_cards=count_cards, count_method=count_method)
    states = (10, 10, 2)  # player sum (12 to 21 possible), dealer card (2-10 + A), no actions

    states_usable_ace = np.zeros(states)
    states_usable_ace_count = np.zeros(states)

    states_no_usable_ace = np.zeros(states)
    states_no_usable_ace_count = np.zeros(states)

    G = 0
    for _ in tqdm(range(n_episodes)):

        # generate random state
        # dealer_shows = np.random.choice(10) + 1
        # player_hand_1 = np.random.choice(10) + 1
        # player_hand_2 = np.random.choice(10) +1
        # p_s = card_value(player_hand_1) + card_value(player_hand_2)
        # player_hand = [player_hand_1, player_hand_2]
        # if p_s == 22:
        #     p_s = 12
        # while p_s < 12:
        #     player_hand.append(card_value(np.random.choice(10) + 1))
        #     p_s += player_hand[-1]
        # player_ace = 1 in player_hand
        # index = np.random.choice(5)
        # state = (player_ace, p_s, dealer_shows, index, player_hand)
        # random state should be used here, but game is random enough, so only random actions,
        # random states was too buggy and couldnt be fixed in time
        episode = bj.play(policy)  # generate episode

        for i, ((ace, player_sum, dealer, _, _), action, reward) in enumerate(reversed(episode)):

            # calculate array index
            if player_sum > 11:
                row = player_sum - 12
            else:
                row = player_sum
            column = dealer - 1

            assert 0 <= row < 10
            assert 0 <= column < 10

            G = gamma * G + reward

            if ace:
                states_usable_ace_count[row, column, action] += 1
                states_usable_ace[row, column, action] += G  # no append, do average by division
            else:
                states_no_usable_ace_count[row, column, action] += 1
                states_no_usable_ace[row, column, action] += G

    Q_ace = np.divide(states_usable_ace, states_usable_ace_count,
                      out=np.zeros_like(states_usable_ace),
                      where=states_usable_ace_count != 0)  # prevent division by zero
    Q_no_ace = np.divide(states_no_usable_ace, states_no_usable_ace_count,
                         out=np.zeros_like(states_no_usable_ace),
                         where=states_no_usable_ace_count != 0)

    pi_ace = np.argmax(Q_ace, axis=2)
    pi_no_ace = np.argmax(Q_no_ace, axis=2)

    return Q_ace, Q_no_ace, pi_ace, pi_no_ace



#states_usable_ace, states_no_usable_ace, policy_ace, policy_no_ace = MC_ES(10000, random_action, 0)


