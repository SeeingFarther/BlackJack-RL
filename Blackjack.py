import random

# Deck
new_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 4
card_to_value = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10,
                 "A": 11}
value_to_card = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K",
                 14: "A"}

# Scores
LOSING_SCORE = -1
WINNING_SCORE = 1
TIE_SCORE = 0


class Hand:
    def __init__(self):
        self.hand = None

    # Sum of the cards in the hand
    def sumHand(self):
        total = 0
        for card in self.hand:
            total += card_to_value[card]
        return total

    # Deal cards
    def deal(self, deck):
        self.hand = []

        # Hand is empty or got served with cards exceeding the result?
        while len(self.hand) == 0:
            for i in range(2):
                random.shuffle(deck)
                card = deck.pop()
                self.hand.append(value_to_card[card])

            # Got cards with sum less than 21 else re-deal
            if card_to_value[self.hand[0]] + card_to_value[self.hand[1]] > 21:
                deck.append(card_to_value[self.hand[0]])
                deck.append(card_to_value[self.hand[1]])
                self.hand = []

    # Hit hand
    def hitHand(self, deck):
        # Deal another card
        card = deck.pop()
        self.hand.append(value_to_card[card])
        return

    # Get the first card of the hand
    def getFirstCard(self):
        if self.hand:
            return card_to_value[self.hand[0]]
        return None


class BlackJack:
    def __init__(self):
        self.deck = None
        self.player_hand = Hand()
        self.dealer_hand = Hand()

    def start_game(self):
        # Init deck
        self.deck = list(new_deck)
        random.shuffle(self.deck)

        # Deal players
        self.player_hand.deal(self.deck)
        self.dealer_hand.deal(self.deck)
        return self.getState()

    # Hit player
    def hitPlayer(self):
        self.player_hand.hitHand(self.deck)
        return self.getState()

    # Hit dealer
    def standPlayer(self):
        while self.dealer_hand.sumHand() < 16:
            self.dealer_hand.hitHand(self.deck)
        return

    # Calculate score
    def score(self):
        player_score = self.player_hand.sumHand()
        dealer_score = self.dealer_hand.sumHand()

        if player_score > 21:
            return LOSING_SCORE
        if dealer_score > 21:
            return WINNING_SCORE
        if player_score == dealer_score:
            return TIE_SCORE
        if player_score == 21:
            return WINNING_SCORE
        if dealer_score == 21:
            return LOSING_SCORE
        if dealer_score < player_score:
            return WINNING_SCORE
        if dealer_score > player_score:
            return LOSING_SCORE

    # Get player score
    def getPlayerScore(self):
        return self.player_hand.sumHand()

    # Get dealer score
    def getDealerScore(self):
        return self.dealer_hand.sumHand()

    # Get the state
    def getState(self):
        sum = self.player_hand.sumHand()
        return [sum, self.dealer_hand.getFirstCard()]
