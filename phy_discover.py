import random


# ============================================================
# PHYDISCOVER V1
#
# Attempts to rediscover simple physics equations using
# evolutionary symbolic regression.
# ============================================================

random.seed(42)


# ============================================================
# 1. Generate physics data
# ============================================================

def make_data():

    data = {
        "F_ma": [],
        "v_at": [],
        "x_at": []
    }

    # F = ma
    for _ in range(30):

        m = random.uniform(1, 10)
        a = random.uniform(0.5, 10)

        F = m * a

        data["F_ma"].append({
            "m": m,
            "a": a,
            "F": F
        })

    # v = at
    for _ in range(30):

        a = random.uniform(0.5, 10)
        t = random.uniform(0.1, 10)

        v = a * t

        data["v_at"].append({
            "a": a,
            "t": t,
            "v": v
        })

    # x = 1/2 at²
    for _ in range(30):

        a = random.uniform(0.5, 10)
        t = random.uniform(0.1, 10)

        x = 0.5 * a * t ** 2

        data["x_at"].append({
            "a": a,
            "t": t,
            "x": x
        })

    return data


# ============================================================
# 2. Expression
# ============================================================

class Expression:

    def __init__(
        self,
        value=None,
        left=None,
        op=None,
        right=None
    ):

        self.value = value
        self.left = left
        self.op = op
        self.right = right

    # --------------------------------------------------------
    # Is this expression just a variable/number?
    # --------------------------------------------------------

    def is_leaf(self):

        return self.value is not None

    # --------------------------------------------------------
    # Evaluate expression
    # --------------------------------------------------------

    def evaluate(self, variables):

        # Leaf node
        if self.is_leaf():

            # Variable
            if isinstance(self.value, str):

                # .get() prevents KeyError
                return variables.get(self.value)

            # Number
            return self.value

        # Evaluate left side
        left = self.left.evaluate(variables)

        # Evaluate right side
        right = self.right.evaluate(variables)

        # Invalid expression
        if left is None or right is None:
            return None

        try:

            if self.op == "+":
                return left + right

            elif self.op == "-":
                return left - right

            elif self.op == "*":
                return left * right

            elif self.op == "/":

                if abs(right) < 1e-10:
                    return None

                return left / right

        except (
            ValueError,
            ZeroDivisionError,
            OverflowError
        ):

            return None

        return None

    # --------------------------------------------------------
    # Convert expression to readable text
    # --------------------------------------------------------

    def __str__(self):

        if self.is_leaf():

            if isinstance(self.value, float):

                return str(round(self.value, 3))

            return str(self.value)

        return (
            f"({self.left} "
            f"{self.op} "
            f"{self.right})"
        )


# ============================================================
# 3. Random expression generation
# ============================================================

VARIABLES = [
    "m",
    "a",
    "t"
]

CONSTANTS = [
    0.5,
    1,
    2,
    3,
    9.8
]

OPERATORS = [
    "+",
    "-",
    "*",
    "/"
]


def random_leaf():

    # Usually choose a variable
    if random.random() < 0.7:

        return Expression(
            value=random.choice(VARIABLES)
        )

    return Expression(
        value=random.choice(CONSTANTS)
    )


def random_expression(depth=0):

    # Stop recursion
    if depth >= 3:

        return random_leaf()

    # Sometimes create a leaf
    if random.random() < 0.35:

        return random_leaf()

    left = random_expression(depth + 1)
    right = random_expression(depth + 1)

    operator = random.choice(OPERATORS)

    return Expression(
        left=left,
        op=operator,
        right=right
    )


# ============================================================
# 4. Score an expression
# ============================================================

def score_expression(
    expression,
    dataset,
    target
):

    total_error = 0
    valid_predictions = 0

    for row in dataset:

        prediction = expression.evaluate(row)

        # Invalid equation
        if prediction is None:
            continue

        actual = row[target]

        denominator = max(
            abs(actual),
            1e-8
        )

        relative_error = (
            abs(prediction - actual)
            / denominator
        )

        total_error += relative_error

        valid_predictions += 1

    # Equation doesn't work
    if valid_predictions == 0:

        return float("inf")

    average_error = (
        total_error
        / valid_predictions
    )

    # Slight complexity penalty
    complexity_penalty = (
        len(str(expression))
        * 0.0001
    )

    return (
        average_error
        + complexity_penalty
    )


# ============================================================
# 5. Mutation
# ============================================================

def mutate(expression):

    # Occasionally completely replace it
    if random.random() < 0.15:

        return random_expression()

    # Leaf
    if expression.is_leaf():

        if random.random() < 0.5:

            return random_leaf()

        return expression

    # Mutate left side
    if random.random() < 0.5:

        return Expression(
            left=mutate(expression.left),
            op=expression.op,
            right=expression.right
        )

    # Mutate right side
    return Expression(
        left=expression.left,
        op=expression.op,
        right=mutate(expression.right)
    )


# ============================================================
# 6. Evolutionary discovery
# ============================================================

def discover(
    dataset,
    target,
    generations=300
):

    population_size = 300

    population = [
        random_expression()
        for _ in range(population_size)
    ]

    best_expression = None
    best_score = float("inf")

    for generation in range(generations):

        scored = []

        for expression in population:

            score = score_expression(
                expression,
                dataset,
                target
            )

            scored.append(
                (score, expression)
            )

            # New best
            if score < best_score:

                best_score = score
                best_expression = expression

        # Sort best → worst
        scored.sort(
            key=lambda item: item[0]
        )

        # Keep the best 10%
        survivors = [
            expression
            for _, expression
            in scored[:30]
        ]

        # Progress
        if generation % 25 == 0:

            print(
                f"Generation {generation:3} | "
                f"error = {best_score:.8f} | "
                f"{best_expression}"
            )

        # New population
        population = survivors.copy()

        while len(population) < population_size:

            parent = random.choice(
                survivors
            )

            child = mutate(parent)

            population.append(child)

    return (
        best_expression,
        best_score
    )


# ============================================================
# 7. Main program
# ============================================================

def main():

    data = make_data()

    experiments = [

        (
            "F = ?",
            data["F_ma"],
            "F"
        ),

        (
            "v = ?",
            data["v_at"],
            "v"
        ),

        (
            "x = ?",
            data["x_at"],
            "x"
        )

    ]

    print()
    print("=" * 60)
    print("             PHYDISCOVER V1")
    print("=" * 60)
    print()

    for name, dataset, target in experiments:

        print()
        print("-" * 60)
        print(
            f"Searching for: {name}"
        )
        print("-" * 60)

        result, score = discover(
            dataset,
            target
        )

        print()
        print("DISCOVERED:")
        print(
            f"    {target} = {result}"
        )

        print(
            f"ERROR: {score:.10f}"
        )

    print()
    print("=" * 60)
    print("Search complete.")
    print("=" * 60)


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    main()
