import random
import math

# ============================================================
# PHYDISCOVER V1
# Attempts to rediscover simple physics equations
# using symbolic regression + evolution.
# ============================================================

random.seed(42)

# ------------------------------------------------------------
# 1. Generate physics data
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. Mathematical expression system
# ------------------------------------------------------------

VARIABLES = ["m", "a", "t"]

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


class Expression:

    def __init__(self, value=None, left=None, op=None, right=None):
        self.value = value
        self.left = left
        self.op = op
        self.right = right

    def is_leaf(self):
        return self.value is not None

    def evaluate(self, variables):

        if self.is_leaf():

            if isinstance(self.value, str):
                return variables[self.value]

            return self.value

        left = self.left.evaluate(variables)
        right = self.right.evaluate(variables)

        try:

            if self.op == "+":
                return left + right

            if self.op == "-":
                return left - right

            if self.op == "*":
                return left * right

            if self.op == "/":

                if abs(right) < 1e-10:
                    return None

                return left / right

        except:
            return None

    def __str__(self):

        if self.is_leaf():

            if isinstance(self.value, float):
                return str(round(self.value, 3))

            return self.value

        return f"({self.left} {self.op} {self.right})"


# ------------------------------------------------------------
# 3. Random expression generator
# ------------------------------------------------------------

def random_leaf():

    if random.random() < 0.7:
        return Expression(value=random.choice(VARIABLES))

    return Expression(value=random.choice(CONSTANTS))


def random_expression(depth=0):

    # Stop growing the tree
    if depth >= 3 or random.random() < 0.35:
        return random_leaf()

    left = random_expression(depth + 1)
    right = random_expression(depth + 1)

    op = random.choice(OPERATORS)

    return Expression(
        left=left,
        op=op,
        right=right
    )


# ------------------------------------------------------------
# 4. Score an equation
# ------------------------------------------------------------

def score_expression(expr, dataset, target):

    error = 0
    valid = 0

    for row in dataset:

        prediction = expr.evaluate(row)

        if prediction is None:
            continue

        actual = row[target]

        # Relative error
        denominator = max(abs(actual), 1e-8)

        relative_error = abs(prediction - actual) / denominator

        error += relative_error
        valid += 1

    if valid == 0:
        return float("inf")

    # Slight penalty for complicated equations
    complexity = len(str(expr)) * 0.0001

    return error / valid + complexity


# ------------------------------------------------------------
# 5. Mutation
# ------------------------------------------------------------

def mutate(expr):

    # Completely replace occasionally
    if random.random() < 0.15:
        return random_expression()

    if expr.is_leaf():

        if random.random() < 0.5:
            return random_leaf()

        return expr

    # Mutate one side
    if random.random() < 0.5:

        return Expression(
            left=mutate(expr.left),
            op=expr.op,
            right=expr.right
        )

    return Expression(
        left=expr.left,
        op=expr.op,
        right=mutate(expr.right)
    )


# ------------------------------------------------------------
# 6. Evolutionary search
# ------------------------------------------------------------

def discover(dataset, target, generations=300):

    population_size = 300

    population = [
        random_expression()
        for _ in range(population_size)
    ]

    best = None
    best_score = float("inf")

    for generation in range(generations):

        scored = []

        for expr in population:

            score = score_expression(
                expr,
                dataset,
                target
            )

            scored.append((score, expr))

            if score < best_score:

                best_score = score
                best = expr

        scored.sort(key=lambda x: x[0])

        # Keep best 10%
        survivors = [
            expr
            for _, expr in scored[:30]
        ]

        # Print progress
        if generation % 25 == 0:

            print(
                f"Generation {generation:3} | "
                f"error = {best_score:.8f} | "
                f"{best}"
            )

        # Create next generation
        population = survivors.copy()

        while len(population) < population_size:

            parent = random.choice(survivors)

            child = mutate(parent)

            population.append(child)

    return best, best_score


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main():

    data = make_data()

    experiments = [
        ("F = ?", data["F_ma"], "F"),
        ("v = ?", data["v_at"], "v"),
        ("x = ?", data["x_at"], "x")
    ]

    print()
    print("=" * 60)
    print("             PHYDISCOVER V1")
    print("=" * 60)
    print()

    for name, dataset, target in experiments:

        print()
        print("-" * 60)
        print(f"Searching for: {name}")
        print("-" * 60)

        result, score = discover(
            dataset,
            target
        )

        print()
        print("DISCOVERED:")
        print(f"    {target} = {result}")

        print(f"ERROR: {score:.10f}")

    print()
    print("=" * 60)
    print("Search complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
