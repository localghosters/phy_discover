import random
import math
import copy

# ============================================================
# PHYDISCOVER V2.1
# FIXED VERSION
# ============================================================

random.seed(42)


# ============================================================
# 1. DATA
# ============================================================

def make_data():

    data = {
        "F_ma": [],
        "v_at": [],
        "x_at": [],
        "p_mv": [],
        "E_mgh": [],
        "KE": []
    }

    # --------------------------------------------------------
    # Newton:
    # F = ma
    # --------------------------------------------------------

    for _ in range(100):

        m = random.uniform(1, 10)
        a = random.uniform(0.5, 10)

        F = m * a

        data["F_ma"].append({
            "m": m,
            "a": a,
            "F": F
        })

    # --------------------------------------------------------
    # Kinematics:
    # v = at
    # --------------------------------------------------------

    for _ in range(100):

        a = random.uniform(0.5, 10)
        t = random.uniform(0.1, 10)

        v = a * t

        data["v_at"].append({
            "a": a,
            "t": t,
            "v": v
        })

    # --------------------------------------------------------
    # Kinematics:
    # x = 1/2 at²
    # --------------------------------------------------------

    for _ in range(100):

        a = random.uniform(0.5, 10)
        t = random.uniform(0.1, 10)

        x = 0.5 * a * t ** 2

        data["x_at"].append({
            "a": a,
            "t": t,
            "x": x
        })

    # --------------------------------------------------------
    # Momentum:
    # p = mv
    # --------------------------------------------------------

    for _ in range(100):

        m = random.uniform(1, 10)
        v = random.uniform(0.5, 20)

        p = m * v

        data["p_mv"].append({
            "m": m,
            "v": v,
            "p": p
        })

    # --------------------------------------------------------
    # Potential energy:
    # E = mgh
    # --------------------------------------------------------

    for _ in range(100):

        m = random.uniform(1, 10)
        g = 9.8
        h = random.uniform(0.1, 20)

        E = m * g * h

        data["E_mgh"].append({
            "m": m,
            "g": g,
            "h": h,
            "E": E
        })

    # --------------------------------------------------------
    # Kinetic energy:
    # KE = 1/2 mv²
    # --------------------------------------------------------

    for _ in range(100):

        m = random.uniform(1, 10)
        v = random.uniform(0.5, 20)

        KE = 0.5 * m * v ** 2

        data["KE"].append({
            "m": m,
            "v": v,
            "KE": KE
        })

    return data


# ============================================================
# 2. NODE
# ============================================================

class Node:

    UNARY_OPS = {
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "abs",
        "neg"
    }

    def __init__(
        self,
        value=None,
        op=None,
        left=None,
        right=None
    ):

        self.value = value
        self.op = op
        self.left = left
        self.right = right

    # --------------------------------------------------------
    # Leaf?
    # --------------------------------------------------------

    def is_leaf(self):

        return self.op is None

    # --------------------------------------------------------
    # Deep copy
    # --------------------------------------------------------

    def clone(self):

        return copy.deepcopy(self)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    def evaluate(self, variables):

        if self.is_leaf():

            if isinstance(self.value, str):

                return variables.get(self.value)

            return self.value

        try:

            # ------------------------------------------------
            # Unary operation
            # ------------------------------------------------

            if self.op in Node.UNARY_OPS:

                if self.left is None:
                    return None

                x = self.left.evaluate(variables)

                if x is None:
                    return None

                if not math.isfinite(x):
                    return None

                if self.op == "sin":
                    return math.sin(x)

                if self.op == "cos":
                    return math.cos(x)

                if self.op == "exp":

                    if x > 50 or x < -50:
                        return None

                    return math.exp(x)

                if self.op == "log":

                    if x <= 0:
                        return None

                    return math.log(x)

                if self.op == "sqrt":

                    if x < 0:
                        return None

                    return math.sqrt(x)

                if self.op == "abs":
                    return abs(x)

                if self.op == "neg":
                    return -x

            # ------------------------------------------------
            # Binary operation
            # ------------------------------------------------

            if self.left is None or self.right is None:
                return None

            left = self.left.evaluate(variables)
            right = self.right.evaluate(variables)

            if left is None or right is None:
                return None

            if not math.isfinite(left):
                return None

            if not math.isfinite(right):
                return None

            if self.op == "+":

                result = left + right

            elif self.op == "-":

                result = left - right

            elif self.op == "*":

                result = left * right

            elif self.op == "/":

                if abs(right) < 1e-12:
                    return None

                result = left / right

            elif self.op == "^":

                # Avoid explosive expressions
                if abs(left) > 1e6:
                    return None

                if abs(right) > 10:
                    return None

                # Negative number to non-integer power
                if (
                    left < 0
                    and abs(right - round(right)) > 1e-10
                ):
                    return None

                result = left ** right

            else:

                return None

            if not math.isfinite(result):
                return None

            return result

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError
        ):

            return None

    # --------------------------------------------------------
    # String representation
    # --------------------------------------------------------

    def __str__(self):

        if self.is_leaf():

            if isinstance(self.value, float):

                return f"{self.value:.6g}"

            return str(self.value)

        if self.op in Node.UNARY_OPS:

            if self.left is None:
                return f"{self.op}(?)"

            return f"{self.op}({self.left})"

        left_string = (
            str(self.left)
            if self.left is not None
            else "?"
        )

        right_string = (
            str(self.right)
            if self.right is not None
            else "?"
        )

        return (
            f"({left_string} "
            f"{self.op} "
            f"{right_string})"
        )

    # --------------------------------------------------------
    # Structural size
    # --------------------------------------------------------

    def size(self):

        if self.is_leaf():
            return 1

        if self.op in Node.UNARY_OPS:

            if self.left is None:
                return 1

            return 1 + self.left.size()

        left_size = (
            self.left.size()
            if self.left is not None
            else 0
        )

        right_size = (
            self.right.size()
            if self.right is not None
            else 0
        )

        return 1 + left_size + right_size

    # --------------------------------------------------------
    # Maximum depth
    # --------------------------------------------------------

    def depth(self):

        if self.is_leaf():
            return 1

        if self.op in Node.UNARY_OPS:

            if self.left is None:
                return 1

            return 1 + self.left.depth()

        left_depth = (
            self.left.depth()
            if self.left is not None
            else 0
        )

        right_depth = (
            self.right.depth()
            if self.right is not None
            else 0
        )

        return 1 + max(
            left_depth,
            right_depth
        )


# ============================================================
# 3. SELF-GROWING GRAMMAR
# ============================================================

class Grammar:

    def __init__(self):

        self.variables = set([
            "m",
            "a",
            "t",
            "v",
            "g",
            "h"
        ])

        self.constants = [
            0.5,
            1.0,
            2.0,
            3.0,
            9.8
        ]

        self.binary_ops = [
            "+",
            "-",
            "*",
            "/"
        ]

        self.unary_ops = []

        self.powers = [
            2.0
        ]

        self.growth_level = 0

    # --------------------------------------------------------
    # Grow grammar
    # --------------------------------------------------------

    def grow(self):

        self.growth_level += 1

        if self.growth_level == 1:

            if "^" not in self.binary_ops:

                self.binary_ops.append("^")

            self.powers.extend([
                0.5,
                3.0
            ])

        elif self.growth_level == 2:

            for op in [
                "sqrt",
                "abs",
                "neg"
            ]:

                if op not in self.unary_ops:

                    self.unary_ops.append(op)

        elif self.growth_level == 3:

            for op in [
                "log",
                "exp"
            ]:

                if op not in self.unary_ops:

                    self.unary_ops.append(op)

        elif self.growth_level == 4:

            for op in [
                "sin",
                "cos"
            ]:

                if op not in self.unary_ops:

                    self.unary_ops.append(op)

        elif self.growth_level >= 5:

            candidates = [
                0.25,
                0.3333333333,
                4.0,
                6.28,
                9.81
            ]

            for c in candidates:

                if c not in self.constants:

                    self.constants.append(c)

        print(
            f"\n*** GRAMMAR GROWN TO LEVEL "
            f"{self.growth_level} ***"
        )


# ============================================================
# 4. RANDOM NODE GENERATION
# ============================================================

def random_leaf(grammar):

    r = random.random()

    if r < 0.65:

        return Node(
            value=random.choice(
                list(grammar.variables)
            )
        )

    return Node(
        value=random.choice(
            grammar.constants
        )
    )


def random_expression(
    grammar,
    depth=0,
    max_depth=4
):

    # Stop recursion
    if depth >= max_depth:

        return random_leaf(grammar)

    # Leaf probability
    if random.random() < 0.30:

        return random_leaf(grammar)

    # Unary operation
    if (
        grammar.unary_ops
        and random.random() < 0.15
    ):

        return Node(
            op=random.choice(
                grammar.unary_ops
            ),
            left=random_expression(
                grammar,
                depth + 1,
                max_depth
            )
        )

    # Binary
    left = random_expression(
        grammar,
        depth + 1,
        max_depth
    )

    right = random_expression(
        grammar,
        depth + 1,
        max_depth
    )

    op = random.choice(
        grammar.binary_ops
    )

    # Power uses a known numerical exponent
    if op == "^":

        right = Node(
            value=random.choice(
                grammar.powers
            )
        )

    return Node(
        op=op,
        left=left,
        right=right
    )


# ============================================================
# 5. RANDOM CONSTANT LEARNING
# ============================================================

def random_constant():

    candidates = [
        0.1,
        0.25,
        0.333333,
        0.5,
        0.666666,
        1,
        2,
        3,
        4,
        5,
        9.8,
        9.81,
        math.pi
    ]

    return random.choice(candidates)


# ============================================================
# 6. CONSTANT MUTATION
# ============================================================

def mutate_constant(node):

    if node is None:
        return Node(value=random_constant())

    if not node.is_leaf():
        return node.clone()

    if isinstance(node.value, str):
        return node.clone()

    if random.random() < 0.5:

        scale = random.uniform(
            0.8,
            1.2
        )

        new_value = (
            node.value * scale
        )

        return Node(
            value=new_value
        )

    return Node(
        value=random_constant()
    )


# ============================================================
# 7. MUTATION
# ============================================================

def mutate(
    node,
    grammar,
    probability=0.20
):

    # --------------------------------------------------------
    # Safety against None
    # --------------------------------------------------------

    if node is None:

        return random_expression(
            grammar,
            max_depth=3
        )

    if random.random() > probability:

        return node.clone()

    # --------------------------------------------------------
    # Completely replace subtree
    # --------------------------------------------------------

    if random.random() < 0.12:

        return random_expression(
            grammar,
            max_depth=3
        )

    # --------------------------------------------------------
    # Leaf mutation
    # --------------------------------------------------------

    if node.is_leaf():

        return mutate_constant(node)

    # --------------------------------------------------------
    # Unary
    # --------------------------------------------------------

    if node.op in Node.UNARY_OPS:

        return Node(
            op=node.op,
            left=mutate(
                node.left,
                grammar,
                probability
            )
        )

    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    if random.random() < 0.20:

        new_op = random.choice(
            grammar.binary_ops
        )

        left = (
            node.left.clone()
            if node.left is not None
            else random_leaf(grammar)
        )

        right = (
            node.right.clone()
            if node.right is not None
            else random_leaf(grammar)
        )

        if new_op == "^":

            right = Node(
                value=random.choice(
                    grammar.powers
                )
            )

        return Node(
            op=new_op,
            left=left,
            right=right
        )

    # --------------------------------------------------------
    # Mutate left
    # --------------------------------------------------------

    if random.random() < 0.5:

        left = mutate(
            node.left,
            grammar,
            probability
        )

        right = (
            node.right.clone()
            if node.right is not None
            else random_leaf(grammar)
        )

        return Node(
            op=node.op,
            left=left,
            right=right
        )

    # --------------------------------------------------------
    # Mutate right
    # --------------------------------------------------------

    left = (
        node.left.clone()
        if node.left is not None
        else random_leaf(grammar)
    )

    right = mutate(
        node.right,
        grammar,
        probability
    )

    return Node(
        op=node.op,
        left=left,
        right=right
    )


# ============================================================
# 8. CROSSOVER
# ============================================================

def crossover(a, b):

    # ========================================================
    # IMPORTANT FIX
    #
    # The old version assumed that a and b were always Node
    # objects. Some recursive paths could pass None.
    #
    # We now explicitly handle None.
    # ========================================================

    if a is None and b is None:
        return None

    if a is None:
        return b.clone() if b is not None else None

    if b is None:
        return a.clone()

    # --------------------------------------------------------
    # Occasionally simply clone
    # --------------------------------------------------------

    if random.random() < 0.10:

        return a.clone()

    # --------------------------------------------------------
    # If either is a leaf, safely exchange subtree
    # --------------------------------------------------------

    if a.is_leaf():

        return b.clone()

    if b.is_leaf():

        return a.clone()

    # --------------------------------------------------------
    # Unary node
    # --------------------------------------------------------

    if a.op in Node.UNARY_OPS:

        # If B is unary, crossover its child.
        if b.op in Node.UNARY_OPS:

            return Node(
                op=a.op,
                left=crossover(
                    a.left,
                    b.left
                )
            )

        # Otherwise use B as possible subtree.
        if random.random() < 0.5:

            return Node(
                op=a.op,
                left=crossover(
                    a.left,
                    b
                )
            )

        return a.clone()

    # --------------------------------------------------------
    # If A is binary but B is unary
    # --------------------------------------------------------

    if b.op in Node.UNARY_OPS:

        if random.random() < 0.5:

            return Node(
                op=a.op,
                left=crossover(
                    a.left,
                    b
                ),
                right=(
                    a.right.clone()
                    if a.right is not None
                    else None
                )
            )

        return Node(
            op=a.op,
            left=(
                a.left.clone()
                if a.left is not None
                else None
            ),
            right=crossover(
                a.right,
                b
            )
        )

    # --------------------------------------------------------
    # Normal binary crossover
    # --------------------------------------------------------

    if random.random() < 0.5:

        new_left = crossover(
            a.left,
            b.left
        )

        new_right = (
            a.right.clone()
            if a.right is not None
            else (
                b.right.clone()
                if b.right is not None
                else None
            )
        )

        return Node(
            op=a.op,
            left=new_left,
            right=new_right
        )

    new_left = (
        a.left.clone()
        if a.left is not None
        else (
            b.left.clone()
            if b.left is not None
            else None
        )
    )

    new_right = crossover(
        a.right,
        b.right
    )

    return Node(
        op=a.op,
        left=new_left,
        right=new_right
    )


# ============================================================
# 9. SIMPLIFICATION
# ============================================================

def simplify(node):

    # --------------------------------------------------------
    # None protection
    # --------------------------------------------------------

    if node is None:

        return Node(value=0)

    if node.is_leaf():

        return node.clone()

    # --------------------------------------------------------
    # Unary
    # --------------------------------------------------------

    if node.op in Node.UNARY_OPS:

        child = simplify(node.left)

        # -(-x) = x
        if (
            node.op == "neg"
            and not child.is_leaf()
            and child.op == "neg"
        ):

            return simplify(
                child.left
            )

        return Node(
            op=node.op,
            left=child
        )

    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    left = simplify(node.left)
    right = simplify(node.right)

    # --------------------------------------------------------
    # Algebraic identities
    # --------------------------------------------------------

    # x + 0
    if (
        node.op == "+"
        and right.is_leaf()
        and right.value == 0
    ):

        return left

    # 0 + x
    if (
        node.op == "+"
        and left.is_leaf()
        and left.value == 0
    ):

        return right

    # x - 0
    if (
        node.op == "-"
        and right.is_leaf()
        and right.value == 0
    ):

        return left

    # x * 1
    if (
        node.op == "*"
        and right.is_leaf()
        and right.value == 1
    ):

        return left

    # 1 * x
    if (
        node.op == "*"
        and left.is_leaf()
        and left.value == 1
    ):

        return right

    # x / 1
    if (
        node.op == "/"
        and right.is_leaf()
        and right.value == 1
    ):

        return left

    # x * 0
    if node.op == "*":

        if (
            left.is_leaf()
            and left.value == 0
        ):

            return Node(value=0)

        if (
            right.is_leaf()
            and right.value == 0
        ):

            return Node(value=0)

    # x ^ 1
    if (
        node.op == "^"
        and right.is_leaf()
        and right.value == 1
    ):

        return left

    # x ^ 0 = 1
    if (
        node.op == "^"
        and right.is_leaf()
        and right.value == 0
    ):

        return Node(value=1)

    return Node(
        op=node.op,
        left=left,
        right=right
    )


# ============================================================
# 10. SCORE
# ============================================================

def score_expression(
    expression,
    dataset,
    target
):

    if expression is None:

        return float("inf")

    total_error = 0.0
    valid = 0

    for row in dataset:

        try:

            prediction = expression.evaluate(
                row
            )

        except Exception:

            prediction = None

        if prediction is None:
            continue

        if not math.isfinite(prediction):
            continue

        actual = row[target]

        denominator = max(
            abs(actual),
            1e-10
        )

        relative_error = (
            abs(prediction - actual)
            / denominator
        )

        total_error += relative_error

        valid += 1

    if valid == 0:

        return float("inf")

    error = (
        total_error / valid
    )

    # Complexity penalty
    complexity = expression.size()

    complexity_penalty = (
        complexity * 0.00005
    )

    # Depth penalty
    depth_penalty = (
        expression.depth() * 0.00002
    )

    return (
        error
        + complexity_penalty
        + depth_penalty
    )


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

def split_data(
    dataset,
    ratio=0.8
):

    shuffled = dataset.copy()

    random.shuffle(shuffled)

    cut = int(
        len(shuffled) * ratio
    )

    return (
        shuffled[:cut],
        shuffled[cut:]
    )


# ============================================================
# 12. DISCOVERY
# ============================================================

def discover(
    dataset,
    target,
    grammar,
    generations=500
):

    population_size = 500

    train, test = split_data(
        dataset
    )

    population = [
        random_expression(
            grammar,
            max_depth=4
        )
        for _ in range(
            population_size
        )
    ]

    archive = {}

    best_expression = None
    best_score = float("inf")

    stagnant_generations = 0

    for generation in range(
        generations
    ):

        scored = []

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        for expression in population:

            expression = simplify(
                expression
            )

            score = score_expression(
                expression,
                train,
                target
            )

            scored.append(
                (
                    score,
                    expression
                )
            )

            key = str(expression)

            if (
                key not in archive
                or score < archive[key]
            ):

                archive[key] = score

            if score < best_score:

                best_score = score

                best_expression = (
                    expression.clone()
                )

                stagnant_generations = 0

            else:

                stagnant_generations += 1

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if best_expression is None:

            best_expression = random_expression(
                grammar,
                max_depth=3
            )

            best_score = score_expression(
                best_expression,
                train,
                target
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        scored.sort(
            key=lambda x: x[0]
        )

        # ----------------------------------------------------
        # Elitism
        # ----------------------------------------------------

        elite_count = max(
            10,
            population_size // 20
        )

        elites = [
            expression.clone()
            for _, expression
            in scored[:elite_count]
        ]

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if generation % 25 == 0:

            test_score = score_expression(
                best_expression,
                test,
                target
            )

            print(
                f"Generation {generation:4d} | "
                f"train={best_score:.10f} | "
                f"test={test_score:.10f}"
            )

            print(
                "     ",
                best_expression
            )

        # ----------------------------------------------------
        # Perfect-ish discovery
        # ----------------------------------------------------

        if best_score < 0.0001:

            break

        # ----------------------------------------------------
        # Grow grammar if stuck
        # ----------------------------------------------------

        if stagnant_generations > 75:

            if grammar.growth_level < 6:

                grammar.grow()

            stagnant_generations = 0

        # ----------------------------------------------------
        # Build next generation
        # ----------------------------------------------------

        new_population = elites.copy()

        # Tournament pool
        pool = [
            expression
            for _, expression
            in scored[:100]
        ]

        # Safety
        if not pool:

            pool = [
                random_expression(
                    grammar,
                    max_depth=3
                )
                for _ in range(20)
            ]

        while len(
            new_population
        ) < population_size:

            parent_a = random.choice(
                pool
            )

            # ------------------------------------------------
            # Crossover
            # ------------------------------------------------

            if random.random() < 0.35:

                parent_b = random.choice(
                    pool
                )

                child = crossover(
                    parent_a,
                    parent_b
                )

            else:

                child = parent_a.clone()

            # ------------------------------------------------
            # If crossover somehow produces None
            # ------------------------------------------------

            if child is None:

                child = random_expression(
                    grammar,
                    max_depth=3
                )

            # ------------------------------------------------
            # Mutation
            # ------------------------------------------------

            child = mutate(
                child,
                grammar,
                probability=0.35
            )

            # ------------------------------------------------
            # Simplify
            # ------------------------------------------------

            child = simplify(
                child
            )

            # ------------------------------------------------
            # Reject absurd trees
            # ------------------------------------------------

            if child.size() > 35:

                continue

            if child.depth() > 15:

                continue

            new_population.append(
                child
            )

        population = new_population

    return (
        best_expression,
        best_score,
        score_expression(
            best_expression,
            test,
            target
        ),
        archive
    )


# ============================================================
# 13. DISCOVERY REPORT
# ============================================================

def report(
    target,
    expression,
    train_error,
    test_error
):

    print()
    print("=" * 70)

    print(
        f"DISCOVERY: {target}"
    )

    print("=" * 70)

    print()

    print(
        f"{target} = {expression}"
    )

    print()

    print(
        f"Training error : "
        f"{train_error:.12f}"
    )

    print(
        f"Testing error  : "
        f"{test_error:.12f}"
    )

    print(
        f"Complexity     : "
        f"{expression.size()}"
    )

    print(
        f"Depth          : "
        f"{expression.depth()}"
    )

    print()


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "        PHYDISCOVER V2"
    )

    print(
        "     SELF-GROWING PHYSICS AI"
    )

    print("=" * 70)

    data = make_data()

    grammar = Grammar()

    experiments = [

        (
            "Force",
            data["F_ma"],
            "F"
        ),

        (
            "Velocity",
            data["v_at"],
            "v"
        ),

        (
            "Position",
            data["x_at"],
            "x"
        ),

        (
            "Momentum",
            data["p_mv"],
            "p"
        ),

        (
            "Potential Energy",
            data["E_mgh"],
            "E"
        ),

        (
            "Kinetic Energy",
            data["KE"],
            "KE"
        )

    ]

    discoveries = {}

    for name, dataset, target in experiments:

        print()
        print("-" * 70)

        print(
            f"DISCOVERING: {name}"
        )

        print("-" * 70)

        result = discover(
            dataset,
            target,
            grammar,
            generations=500
        )

        (
            expression,
            train_error,
            test_error,
            archive
        ) = result

        discoveries[target] = (
            expression,
            train_error,
            test_error
        )

        report(
            target,
            expression,
            train_error,
            test_error
        )

    # --------------------------------------------------------
    # Final discoveries
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "              DISCOVERY ARCHIVE"
    )

    print("=" * 70)

    for target, result in discoveries.items():

        expression = result[0]

        print(
            f"{target} = {expression}"
        )

    print()

    print(
        "Search complete."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
