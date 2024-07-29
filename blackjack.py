import numpy as np
import matplotlib
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from strategies import *



def card_value(card_num_val):
    return 11 if card_num_val == 1 else card_num_val


class Blackjack:

    def __init__(self, num_packs=1, DEALERS_POLICY=None,
                 count_cards=False, count_method=None):
        self.BASE_WIN_REWARD = 1
        self.BASE_DRAW_REWARD = 0
        self.BASE_LOSE_REWARD = -1
        # for simple point count
        self.WIN_REWARD = self.BASE_WIN_REWARD
        self.DRAW_REWARD = self.BASE_DRAW_REWARD
        self.LOSE_REWARD = self.BASE_LOSE_REWARD  # penalty more accurate
        self.num_packs = num_packs

        # set cardcounting
        if count_cards:
            if count_method:
                self.use_new_pack = True
                self.count_cards = count_cards
                self.count_method = count_method
            else:
                raise Exception("if count_cards flag is set, method must be set")
        else:
            self.use_new_pack = False
            self.count_cards = False

        # the memory for card counting
        self.point_total = 0
        self.unseen_cards = 52 * self.num_packs
        self.count_index = 0  # represents the result of the card counting that is used, naming is a bit off but should be good enough

        if DEALERS_POLICY: # ill never use this
            self.DEALERS_POLICY = DEALERS_POLICY
        else:
            # the standard one
            self.DEALERS_POLICY = [DRAW for _ in range(22)]
            for i in range(17, 22):
                self.DEALERS_POLICY[i] = STAND

        self.init_pack()

    # init pack and also card counting variables
    def init_pack(self):
        self.pack = []
        self.point_total = 0
        self.unseen_cards = 52 * self.num_packs
        self.count_index = 0
        for _ in range(self.num_packs):
            self.pack.extend([1, 1, 1, 1,
                              2, 2, 2, 2,
                              3, 3, 3, 3,
                              4, 4, 4, 4,
                              5, 5, 5, 5,
                              6, 6, 6, 6,
                              7, 7, 7, 7,
                              8, 8, 8, 8,
                              9, 9, 9, 9,
                              10, 10, 10, 10,
                              10, 10, 10, 10,
                              10, 10, 10, 10,
                              10, 10, 10, 10])


    def deal_card(self, count, explicit_card=None):
        _count = count
        # make sure not running out of cards
        if len(self.pack) == 0:
            self.init_pack()
        if explicit_card:
            if self.pack.pop(self.pack.count(explicit_card)):
                _count = True  # in case of initial state, count, it is the open dealers card
                card = self.pack.pop(self.pack.index(explicit_card))
            else:
                card = explicit_card  # card appears out of nothing, should not distort result too much
        else:
            card = self.pack.pop(random.randint(0, len(self.pack) - 1))
        # if active card counting and card should be counted
        if self.count_cards and _count:
            self.count_card(card)
        return card


    def count_card(self, card):
        if card_value(card) > 9:
            self.point_total -= 1
        elif card_value(card) < 7:
            self.point_total += 1

        if self.count_method == "complete":
            self.complete_point_count()
        else:
            if self.count_method != "simple":
                raise Exception("count method must be either \"simple\" or \"complete\" ")


    def complete_point_count(self):
        self.unseen_cards -= 1
        self.count_index = self.point_total / self.unseen_cards


    def play(self, policy_player, initial_state=None):

        # init pack for no counting
        if self.use_new_pack:
            self.init_pack()

        # change "bet" for simple point count system
        if self.count_cards and self.count_method == "simple" and self.point_total > 0:
            self.WIN_REWARD = self.BASE_WIN_REWARD * self.point_total
            self.LOSE_REWARD = self.BASE_LOSE_REWARD * self.point_total

        # initiate variables
        player_sum = 0
        player_ace = False
        player_hand = []
        player_action = 0

        dealer_sum = 0
        dealer_hand = []
        dealer_action = 0

        steps = []

        # burn first card
        self.deal_card(count=False)

        ##### The Deal #####

        # initial state for ES
        if not initial_state:

            # always draw if under 12, cant go bust
            while player_sum < 12:
                player_hand.append(self.deal_card(count=True))
                player_sum = sum([card_value(card) for card in player_hand])
                no_player_usable_aces = player_hand.count(1)

                # if ace, use it as 1 to avoid index out of bounds
                while player_sum > 21 and no_player_usable_aces > 0:
                    player_sum -= 10
                    no_player_usable_aces -= 1

                player_ace = no_player_usable_aces > 0

                assert player_sum < 22

            # dealer takes two cards
            dealer_hand.append(self.deal_card(count=True))

        else:

            player_ace = initial_state[0]
            player_sum = initial_state[1]
            dealer_hand.append(initial_state[2])
            self.count_index = initial_state[3]
            player_hand = initial_state[4]

        dealer_hand.append(self.deal_card(count=False))
        dealer_sum = sum([card_value(card) for card in dealer_hand])
        no_dealer_usable_aces = dealer_hand.count(1)

        # if ace, use it as 1 to avoid busting
        while dealer_sum > 21 and no_dealer_usable_aces > 0:
            dealer_sum -= 10
            no_dealer_usable_aces -= 1

        assert dealer_sum < 22

        # check for natural
        if card_value(player_hand[0]) + card_value(player_hand[1]) == 21:
            if dealer_sum < 21:
                return [
                    [(True, 21, dealer_hand[0], self.count_index, player_hand[:2].copy()), STAND, self.WIN_REWARD]]
            elif dealer_sum == 21:
                return [
                    [(True, 21, dealer_hand[0], self.count_index, player_hand[:2].copy()), STAND, self.DRAW_REWARD]]

        ##### The Draw #####

        # player's turn
        while True:
            state = (player_ace, player_sum, dealer_hand[0], self.count_index, player_hand.copy())

            # decide action
            player_action = policy_player(state)

            if player_action == DRAW:

                # new card, calculate hand
                player_hand.append(self.deal_card(count=True))
                player_sum = sum([card_value(card) for card in player_hand])
                no_player_usable_aces = player_hand.count(1)

                # if ace, use it as 1 to avoid busting
                while player_sum > 21 and no_player_usable_aces > 0:
                    player_sum -= 10
                    no_player_usable_aces -= 1

                # player busts
                if player_sum > 21:
                    steps.append([state, player_action, self.LOSE_REWARD])
                    return steps

                # make sure no error
                assert player_sum <= 21

                player_ace = no_player_usable_aces > 0

                # track player's trajectory
                steps.append([state, player_action, self.DRAW_REWARD])

            else:
                break

        if self.count_cards:
            self.count_card(dealer_hand[1])  # count the dealers second card, that will now be turned face up

        # dealer's turn
        while True:
            # decide action
            dealer_action = self.DEALERS_POLICY[dealer_sum]

            if dealer_action == DRAW:

                # new card, calculate hand
                dealer_hand.append(self.deal_card(count=True))
                dealer_sum = sum([card_value(card) for card in dealer_hand])
                no_dealer_usable_aces = dealer_hand.count(1)

                # if ace, use it as 1 to avoid busting
                while dealer_sum > 21 and no_dealer_usable_aces > 0:
                    dealer_sum -= 10
                    no_dealer_usable_aces -= 1

                # dealer busts
                if dealer_sum > 21:
                    steps.append([state, player_action, self.WIN_REWARD])
                    return steps

                # make sure no error
                assert dealer_sum <= 21

            else:
                break

        # compare sums player / dealer
        if player_sum > dealer_sum:
            steps.append([state, player_action, self.WIN_REWARD])

        elif player_sum == dealer_sum:
            steps.append([state, player_action, self.DRAW_REWARD])

        else:
            steps.append([state, player_action, self.LOSE_REWARD])

        return steps
