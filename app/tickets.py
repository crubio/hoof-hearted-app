from dataclasses import dataclass, field


@dataclass
class WinBet:
    race: int
    horse: int
    amount: float

    def cost(self) -> float:
        # Approximation only — actual payout depends on final pool odds at post time.
        return self.amount


@dataclass
class Pick3Ticket:
    """
    A Pick 3 ticket covering 3 consecutive races.

    Args:
        races: List of 3 race numbers, e.g. [1, 2, 3]
        legs: List of 3 lists of horse numbers, e.g. [[6, 3], [2], [3, 1]]
        amount: Base bet amount per combination (typically $1.00 or $3.00)

    Note:
        cost() returns the amount wagered, not the expected payout. Actual
        returns are pari-mutuel and determined by the final pool at post time.
    """
    races: list[int]
    legs: list[list[int]]
    amount: float = 1.0

    def cost(self) -> float:
        # Approximation — reflects dollars wagered, not guaranteed payout.
        combos = 1
        for leg in self.legs:
            combos *= len(leg)
        return combos * self.amount


@dataclass
class Pick5Ticket:
    """
    A Pick 5 ticket covering 5 consecutive races.

    Args:
        races: List of 5 race numbers, e.g. [1, 2, 3, 4, 5]
        legs: List of 5 lists of horse numbers, e.g. [[6, 3], [2], [3, 1], [5, 4, 2], [1, 2]]
        amount: Base bet amount per combination (typically $0.50)

    Note:
        cost() returns the amount wagered, not the expected payout. Actual
        returns are pari-mutuel and determined by the final pool at post time.
    """
    races: list[int]
    legs: list[list[int]]
    amount: float = 0.50

    def cost(self) -> float:
        # Approximation — reflects dollars wagered, not guaranteed payout.
        combos = 1
        for leg in self.legs:
            combos *= len(leg)
        return combos * self.amount


@dataclass
class Pick6Ticket:
    """
    A Pick 6 ticket covering 6 consecutive races.

    Args:
        races: List of 6 race numbers, e.g. [4, 5, 6, 7, 8, 9]
        legs: List of 6 lists of horse numbers
        amount: Base bet amount per combination (typically $0.20 or $2.00)

    Note:
        cost() returns the amount wagered, not the expected payout. Actual
        returns are pari-mutuel and determined by the final pool at post time.
    """
    races: list[int]
    legs: list[list[int]]
    amount: float = 0.20

    def cost(self) -> float:
        # Approximation — reflects dollars wagered, not guaranteed payout.
        combos = 1
        for leg in self.legs:
            combos *= len(leg)
        return combos * self.amount
