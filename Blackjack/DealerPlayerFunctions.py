import gym
import numpy as np
from gym.envs.toy_text.blackjack import draw_card, sum_hand, usable_ace, cmp, is_bust, deck

env = gym.make('Blackjack-v1', natural=False, sab=True)


# ------------------------------------------------------------------------------------------ #
# Player

def dealer_state_map(dealer_hand):
    ''' Gives an index 0 to 31 for each hand '''
    # 32 states
    # 2-22 goes to state 0---20
    # Ace, 1-10 with Ace to 21--31
    if not usable_ace(dealer_hand):
        return min(sum_hand(dealer_hand), 22) - 2
    else:
        return sum_hand(dealer_hand) + 10


def state_dealer_map(dealer_state):
    ''' Gives a hand for each state index '''
    if dealer_state <= 20:
        return [dealer_state + 2]
    if dealer_state > 20:
        return [dealer_state - 21, 1]


# Get transition probabilities for the dealer
# Row: current state
# Column: next state
# N.b. Dealer takes a hit until above 17
dealer_states = [i for i in range(32)]
P_dealer = np.zeros((32, 32))
for s in dealer_states:
    for d in deck:
        dealer_deck = state_dealer_map(s)
        if sum_hand(dealer_deck) < 17:
            dealer_deck.append(d)
            ns = dealer_state_map(dealer_deck)
        else:
            ns = s
        P_dealer[s][ns] += 1 / 13

# Probability of 17-21, 18-21, 19-21, 20-21, and 21
# h[k][s] = probability of score between k and 21 if dealer initially has s
# The probabilities are calculated using the property:
# P(k|s) = \sum_s^{\prime} P(s^{prime}|s)P(k|s^{\prime},s)
#        = \sum_s^{\prime} P(s^{prime}|s)P(k|s^{\prime}) (Markov property)
h = dict()
for k in [17, 18, 19, 20, 21]:
    H = np.identity(32)
    v = np.zeros(32)
    for s in dealer_states:
        if sum_hand(state_dealer_map(s)) in range(k, 21 + 1):
            v[s] = 1.
        else:
            H[s] -= P_dealer[s]
    h[k] = np.linalg.solve(H, v)


# Function to give the expected reward from player stick in any state.
def expected_reward(player_sum, dealer_card, h):
    dealer_state = dealer_state_map([dealer_card])
    dealer_bust_prob = 1 - h[17][dealer_state]
    if player_sum < 17:
        return 1. * dealer_bust_prob - 1. * h[17][dealer_state]
    elif player_sum > 21:
        return -1.
    elif player_sum == 21:
        return 1. * (1. - h[21][dealer_state])
    else:
        loss_prob = h[player_sum + 1][dealer_state]
        draw_prob = h[player_sum][dealer_state] - h[player_sum + 1][dealer_state]
        win_prob = 1. - loss_prob - draw_prob
        return 1. * win_prob - 1. * loss_prob


# ----------------------------------------------------------------------------- #
# Player

def player_state_map(state):
    # converts state to a index (int) between 0 and 289
    # dealer: 1, player: 4-22 + player: 12-21
    # dealer: 2, player: 4-22 + player: 12-21
    # ......

    player_sum = state[0]
    dealer_card = state[1]
    ace = state[2]

    if not ace:
        player_hand = min(player_sum, 22) - 4
    else:
        player_hand = player_sum + 7

    return player_hand + 29 * (dealer_card - 1)


def state_player_map(s):
    # converts index back to int
    dealer_card = 1 + s // 29
    player_state = s % 29
    if player_state > 18:
        player_ace = True
        player_sum = player_state - 7
    else:
        player_ace = False
        player_sum = player_state + 4

    return (player_sum, dealer_card, player_ace)


def player_hand(full_state):
    # gets players 'hand' from state
    (player_sum, _, player_ace) = full_state
    if player_ace:
        return [player_sum - 11, 1]
    else:
        return [player_sum]


# index states and actions
states = [i for i in
          range(player_state_map((21, 10, True)) + 1)]  # 'biggest' hand is (21,10,True); 'lowest' is (4,1,False)
nS = len(states)
actions = [0, 1]
nA = len(actions)

# Work out all transition probabilities
P = dict()
for s in states:
    P[s] = dict()
    for a in actions:
        P[s][a] = []

for s in states:
    for a in actions:
        if a:
            # if 'hit' look at all 13 cards that can be dealt and the probs for this.
            for d in deck:
                full_state = state_player_map(s)
                players_hand = player_hand(full_state)  # map to normal format
                p = 1 / 13
                # next state 
                players_hand.append(d)
                next_state = (sum_hand(players_hand), full_state[1], usable_ace(players_hand))
                ns = player_state_map(next_state)
                if is_bust(players_hand):
                    done = True
                    reward = -1.0
                else:
                    done = False
                    reward = 0.0
                P[s][a].append((p, ns, reward, done))
        else:
            # if 'stick' works out expected reward from current state
            full_state = state_player_map(s)
            exp_reward = expected_reward(full_state[0], full_state[1], h)
            P[s][a].append((1., s, exp_reward, True))
