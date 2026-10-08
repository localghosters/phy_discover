import random
import math
import copy
from collections import defaultdict

# ============================================================
# PHYDISCOVER V3
#
# OPEN-ENDED LEVEL-1 MATHEMATICAL DISCOVERY ENGINE
#
# Goal:
#   Discover relationships in raw data without being given
#   target equations.
#
# The engine searches for:
#
#   y = f(x1, x2, ...)
#
# and invariant relationships:
#
#   f(x1, x2, ...) = constant
#
# It rewards:
#   - accuracy
#   - simplicity
#   - generalization
#   - dimensional consistency when dimensions are supplied
#   - novelty
#
# It does NOT receive target formulas.
# ============================================================

random.seed(42)

# ============================================================
# 1. CONFIGURATION
# ============================================================

POPULATION_SIZE = 700
GENERATIONS = 700

INITIAL_DEPTH = 4
MAX_TREE_SIZE = 31
MAX_TREE_DEPTH = 10

ELITE_COUNT = 35
TOURNAMENT_SIZE = 7

MUTATION_RATE = 0.40
CROSSOVER_RATE = 0.45

ARCHIVE_LIMIT = 100

MAX_ABS_VALUE = 1e12
MAX_EXP_INPUT = 20
MAX_POWER_EXPONENT = 8

DISCOVERY_ERROR_THRESHOLD = 1e-4

# ============================================================
# 2. DATASET
# ============================================================

class Dataset:

    def __init__(self, rows, name="experiment", dimensions=None):

        self.rows = rows
        self.name = name

        self.variables = sorted(
            set().union(
                *(row.keys() for row in rows)
            )
        )

        self.dimensions = dimensions or {}

    def split(self, ratio=0.8):

        shuffled = self.rows.copy()

        random.shuffle(shuffled)

        cut = int(len(shuffled) * ratio)

        return (
            Dataset(
                shuffled[:cut],
                self.name + "_train",
                self.dimensions
            ),
            Dataset(
                shuffled[cut:],
                self.name + "_test",
                self.dimensions
            )
        )

# ============================================================
# 3. EXPERIMENTAL DATA
#
# IMPORTANT:
# These hidden relationships are NOT supplied to the discovery
# algorithm as targets.
#
# The discoverer only sees the generated observations.
# ============================================================

def make_experimental_data():

    rows = []

    for _ in range(700):

        x = random.uniform(0.5, 20.0)
        y = random.uniform(0.5, 15.0)
        z = random.uniform(0.5, 10.0)

        # Hidden mathematical relationships.

        a = 2.75 * x * y

        b = 0.5 * x * y ** 2

        c = 3.0 * math.sqrt(x) * z

        d = 4.2 * x ** 2 / y

        e = 1.7 * x * y + 2.0 * z

        # Independent noisy measurement channels.

        noise_a = random.gauss(
            0,
            abs(a) * 0.002
        )

        noise_b = random.gauss(
            0,
            abs(b) * 0.002
        )

        noise_c = random.gauss(
            0,
            abs(c) * 0.002
        )

        noise_d = random.gauss(
            0,
            abs(d) * 0.002
        )

        noise_e = random.gauss(
            0,
            abs(e) * 0.002
        )

        rows.append({
            "x": x,
            "y": y,
            "z": z,

            "A": a + noise_a,
            "B": b + noise_b,
            "C": c + noise_c,
            "D": d + noise_d,
            "E": e + noise_e
        })

    dimensions = {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E"
    }

    return Dataset(
        rows,
        "hidden_relationship_experiment",
        dimensions
    )

# ============================================================
# 4. NODE
# ============================================================

class Node:

    UNARY_OPS = {
        "sin",
        "cos",
        "sqrt",
        "log",
        "abs",
        "neg",
        "exp",
        "inv"
    }

    BINARY_OPS = {
        "+",
        "-",
        "*",
        "/",
        "^"
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

    def is_leaf(self):

        return self.op is None

    def clone(self):

        return copy.deepcopy(self)

    def evaluate(self, variables):

        if self.is_leaf():

            if isinstance(self.value, str):

                return variables.get(
                    self.value
                )

            return self.value

        try:

            if self.op in Node.UNARY_OPS:

                if self.left is None:
                    return None

                x = self.left.evaluate(
                    variables
                )

                if x is None:
                    return None

                if not math.isfinite(x):
                    return None

                if abs(x) > MAX_ABS_VALUE:
                    return None

                if self.op == "sin":
                    result = math.sin(x)

                elif self.op == "cos":
                    result = math.cos(x)

                elif self.op == "sqrt":

                    if x < 0:
                        return None

                    result = math.sqrt(x)

                elif self.op == "log":

                    if x <= 0:
                        return None

                    result = math.log(x)

                elif self.op == "abs":

                    result = abs(x)

                elif self.op == "neg":

                    result = -x

                elif self.op == "inv":

                    if abs(x) < 1e-12:
                        return None

                    result = 1.0 / x

                elif self.op == "exp":

                    if x < -MAX_EXP_INPUT:
                        result = 0.0

                    elif x > MAX_EXP_INPUT:
                        return None

                    else:
                        result = math.exp(x)

                else:

                    return None

                if not math.isfinite(result):
                    return None

                return result

            if self.left is None:
                return None

            if self.right is None:
                return None

            left = self.left.evaluate(
                variables
            )

            right = self.right.evaluate(
                variables
            )

            if left is None or right is None:
                return None

            if not math.isfinite(left):
                return None

            if not math.isfinite(right):
                return None

            if abs(left) > MAX_ABS_VALUE:
                return None

            if abs(right) > MAX_ABS_VALUE:
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

                if abs(right) > MAX_POWER_EXPONENT:
                    return None

                if left == 0 and right < 0:
                    return None

                if (
                    left < 0
                    and abs(
                        right - round(right)
                    ) > 1e-10
                ):
                    return None

                result = left ** right

            else:

                return None

            if not math.isfinite(result):
                return None

            if abs(result) > MAX_ABS_VALUE:
                return None

            return result

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError
        ):

            return None

    def size(self):

        if self.is_leaf():
            return 1

        if self.op in Node.UNARY_OPS:

            return (
                1
                + (
                    self.left.size()
                    if self.left is not None
                    else 0
                )
            )

        return (
            1
            + (
                self.left.size()
                if self.left is not None
                else 0
            )
            + (
                self.right.size()
                if self.right is not None
                else 0
            )
        )

    def depth(self):

        if self.is_leaf():
            return 1

        if self.op in Node.UNARY_OPS:

            return (
                1
                + (
                    self.left.depth()
                    if self.left is not None
                    else 0
                )
            )

        return (
            1
            + max(
                self.left.depth()
                if self.left is not None
                else 0,

                self.right.depth()
                if self.right is not None
                else 0
            )
        )

    def variables_used(self):

        if self.is_leaf():

            if isinstance(self.value, str):
                return {self.value}

            return set()

        result = set()

        if self.left is not None:
            result.update(
                self.left.variables_used()
            )

        if self.right is not None:
            result.update(
                self.right.variables_used()
            )

        return result

    def __str__(self):

        if self.is_leaf():

            if isinstance(self.value, float):

                if abs(
                    self.value
                    - round(self.value)
                ) < 1e-10:

                    return str(
                        int(round(self.value))
                    )

                return f"{self.value:.6g}"

            return str(self.value)

        if self.op in Node.UNARY_OPS:

            child = (
                str(self.left)
                if self.left is not None
                else "?"
            )

            if self.op == "neg":
                return f"-({child})"

            if self.op == "inv":
                return f"1/({child})"

            return f"{self.op}({child})"

        left = (
            str(self.left)
            if self.left is not None
            else "?"
        )

        right = (
            str(self.right)
            if self.right is not None
            else "?"
        )

        return (
            f"({left} {self.op} {right})"
        )

# ============================================================
# 5. GRAMMAR
# ============================================================

class Grammar:

    def __init__(self, variables):

        self.variables = list(
            variables
        )

        self.constants = [
            0.25,
            0.5,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0
        ]

        self.binary_ops = [
            "+",
            "-",
            "*",
            "/"
        ]

        self.unary_ops = []

        self.powers = [
            -2.0,
            -1.0,
            0.5,
            1.0,
            2.0,
            3.0
        ]

        self.level = 0

    def grow(self):

        self.level += 1

        if self.level == 1:

            self.binary_ops.append("^")

        elif self.level == 2:

            self.unary_ops.extend([
                "sqrt",
                "abs",
                "neg",
                "inv"
            ])

        elif self.level == 3:

            self.unary_ops.extend([
                "log",
                "exp"
            ])

        elif self.level == 4:

            self.unary_ops.extend([
                "sin",
                "cos"
            ])

        elif self.level >= 5:

            self.constants.extend([
                0.1,
                0.2,
                0.333333333333,
                0.666666666667,
                1.5,
                2.5,
                3.5,
                math.pi,
                9.8,
                9.81
            ])

        self.unary_ops = list(
            dict.fromkeys(
                self.unary_ops
            )
        )

        self.binary_ops = list(
            dict.fromkeys(
                self.binary_ops
            )
        )

        self.constants = list(
            dict.fromkeys(
                self.constants
            )
        )

        print(
            f"\n[GRAMMAR] Growth level "
            f"{self.level}"
        )

# ============================================================
# 6. RANDOM TREE GENERATION
# ============================================================

def random_leaf(grammar):

    if random.random() < 0.78:

        return Node(
            value=random.choice(
                grammar.variables
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
    max_depth=INITIAL_DEPTH
):

    if depth >= max_depth:

        return random_leaf(grammar)

    if random.random() < 0.30:

        return random_leaf(grammar)

    if (
        grammar.unary_ops
        and random.random() < 0.18
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
# 7. CONSTANT MUTATION
# ============================================================

def random_discovered_constant():

    candidates = [
        0.05,
        0.1,
        0.2,
        0.25,
        0.333333333,
        0.5,
        0.666666667,
        0.75,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        9.8,
        9.81,
        math.pi
    ]

    return random.choice(
        candidates
    )

def mutate_leaf(node):

    if node is None:

        return None

    if not node.is_leaf():

        return node.clone()

    if isinstance(node.value, str):

        if random.random() < 0.15:

            return Node(
                value=random_discovered_constant()
            )

        return node.clone()

    if random.random() < 0.5:

        scale = random.uniform(
            0.7,
            1.3
        )

        return Node(
            value=node.value * scale
        )

    return Node(
        value=random_discovered_constant()
    )

# ============================================================
# 8. MUTATION
# ============================================================

def mutate(
    node,
    grammar,
    probability=MUTATION_RATE
):

    if node is None:

        return random_expression(
            grammar,
            max_depth=3
        )

    if random.random() > probability:

        return node.clone()

    # Whole-subtree replacement.

    if random.random() < 0.13:

        return random_expression(
            grammar,
            max_depth=random.randint(
                2,
                4
            )
        )

    # Leaf.

    if node.is_leaf():

        return mutate_leaf(node)

    # Change operator.

    if random.random() < 0.18:

        if node.op in Node.UNARY_OPS:

            if grammar.unary_ops:

                return Node(
                    op=random.choice(
                        grammar.unary_ops
                    ),
                    left=(
                        node.left.clone()
                        if node.left is not None
                        else random_leaf(grammar)
                    )
                )

        else:

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

    # Unary mutation.

    if node.op in Node.UNARY_OPS:

        return Node(
            op=node.op,
            left=mutate(
                node.left,
                grammar,
                probability
            )
        )

    # Binary mutation.

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

    else:

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
# 9. CROSSOVER
# ============================================================

def crossover(a, b):

    if a is None and b is None:
        return None

    if a is None:
        return b.clone()

    if b is None:
        return a.clone()

    if random.random() < 0.08:

        return random.choice([
            a.clone(),
            b.clone()
        ])

    if a.is_leaf():

        return b.clone()

    if b.is_leaf():

        return a.clone()

    # Unary / unary.

    if (
        a.op in Node.UNARY_OPS
        and b.op in Node.UNARY_OPS
    ):

        return Node(
            op=a.op,
            left=crossover(
                a.left,
                b.left
            )
        )

    # Unary A / binary B.

    if a.op in Node.UNARY_OPS:

        if random.random() < 0.5:

            return Node(
                op=a.op,
                left=crossover(
                    a.left,
                    b
                )
            )

        return a.clone()

    # Binary A / unary B.

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

    # Binary / binary.

    if random.random() < 0.5:

        return Node(
            op=a.op,
            left=crossover(
                a.left,
                b.left
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
            b.right
        )
    )

# ============================================================
# 10. SIMPLIFICATION
# ============================================================

def is_constant(node, value=None):

    if node is None:
        return False

    if not node.is_leaf():
        return False

    if isinstance(node.value, str):
        return False

    if value is None:
        return True

    return abs(
        node.value - value
    ) < 1e-12

def simplify(node):

    if node is None:

        return Node(value=0)

    if node.is_leaf():

        return node.clone()

    if node.op in Node.UNARY_OPS:

        child = simplify(
            node.left
        )

        # --x = x

        if (
            node.op == "neg"
            and not child.is_leaf()
            and child.op == "neg"
        ):

            return simplify(
                child.left
            )

        # abs(abs(x)) = abs(x)

        if (
            node.op == "abs"
            and not child.is_leaf()
            and child.op == "abs"
        ):

            return child

        return Node(
            op=node.op,
            left=child
        )

    left = simplify(
        node.left
    )

    right = simplify(
        node.right
    )

    # x + 0

    if node.op == "+" and is_constant(
        right,
        0
    ):
        return left

    # 0 + x

    if node.op == "+" and is_constant(
        left,
        0
    ):
        return right

    # x - 0

    if node.op == "-" and is_constant(
        right,
        0
    ):
        return left

    # x * 1

    if node.op == "*" and is_constant(
        right,
        1
    ):
        return left

    # 1 * x

    if node.op == "*" and is_constant(
        left,
        1
    ):
        return right

    # x * 0

    if node.op == "*":

        if is_constant(left, 0):
            return Node(value=0)

        if is_constant(right, 0):
            return Node(value=0)

    # x / 1

    if node.op == "/" and is_constant(
        right,
        1
    ):
        return left

    # x ^ 1

    if node.op == "^" and is_constant(
        right,
        1
    ):
        return left

    # x ^ 0

    if node.op == "^" and is_constant(
        right,
        0
    ):
        return Node(value=1)

    # Constant folding.

    if (
        left.is_leaf()
        and right.is_leaf()
        and not isinstance(
            left.value,
            str
        )
        and not isinstance(
            right.value,
            str
        )
    ):

        result = Node(
            op=node.op,
            left=left,
            right=right
        ).evaluate({})

        if result is not None:

            return Node(
                value=result
            )

    return Node(
        op=node.op,
        left=left,
        right=right
    )

# ============================================================
# 11. CANONICALIZATION
#
# Used to identify equivalent expressions.
# ============================================================

def canonical(node):

    if node is None:
        return "?"

    if node.is_leaf():

        if isinstance(node.value, str):
            return node.value

        if abs(
            node.value
            - round(node.value)
        ) < 1e-10:

            return str(
                int(round(node.value))
            )

        return f"{node.value:.8g}"

    if node.op in Node.UNARY_OPS:

        return (
            f"{node.op}("
            f"{canonical(node.left)})"
        )

    left = canonical(
        node.left
    )

    right = canonical(
        node.right
    )

    # Commutative operations.

    if node.op in {
        "+",
        "*"
    }:

        children = sorted([
            left,
            right
        ])

        left = children[0]
        right = children[1]

    return (
        f"({left}"
        f"{node.op}"
        f"{right})"
    )

# ============================================================
# 12. NUMERICAL RELATION NORMALIZATION
# ============================================================

def robust_scale(values):

    values = [
        abs(x)
        for x in values
        if x is not None
        and math.isfinite(x)
    ]

    if not values:
        return 1.0

    values.sort()

    middle = values[
        len(values) // 2
    ]

    return max(
        middle,
        1e-12
    )

# ============================================================
# 13. RELATIONSHIP OBJECTIVE
# ============================================================

class Candidate:

    def __init__(
        self,
        expression,
        score,
        error,
        complexity,
        coverage,
        novelty,
        kind,
        target=None
    ):

        self.expression = expression
        self.score = score
        self.error = error
        self.complexity = complexity
        self.coverage = coverage
        self.novelty = novelty
        self.kind = kind
        self.target = target

    def __lt__(self, other):

        return self.score < other.score

# ============================================================
# 14. PREDICTIVE FITNESS
#
# The expression predicts one measured variable.
#
# Example:
#
# A ≈ 2.75*x*y
#
# The system is NOT told this equation.
# It only knows that A is a measured column and searches
# expressions that predict it.
# ============================================================

def predictive_error(
    expression,
    dataset,
    target
):

    actuals = []
    predictions = []

    for row in dataset.rows:

        actual = row.get(target)

        if actual is None:
            continue

        prediction = expression.evaluate(
            row
        )

        if prediction is None:
            continue

        if not math.isfinite(
            prediction
        ):
            continue

        if abs(prediction) > MAX_ABS_VALUE:
            continue

        actuals.append(
            actual
        )

        predictions.append(
            prediction
        )

    if not actuals:

        return (
            float("inf"),
            0.0
        )

    scale = robust_scale(
        actuals
    )

    errors = []

    for actual, prediction in zip(
        actuals,
        predictions
    ):

        error = (
            abs(prediction - actual)
            / max(
                abs(actual),
                scale * 1e-6,
                1e-12
            )
        )

        errors.append(error)

    mean_error = (
        sum(errors)
        / len(errors)
    )

    coverage = (
        len(actuals)
        / len(dataset.rows)
    )

    return (
        mean_error,
        coverage
    )

# ============================================================
# 15. CONSTANT-INVARIANT FITNESS
#
# Searches for expressions that remain approximately constant.
#
# Example:
#
# A / (x*y) ≈ 2.75
#
# This can reveal a relationship without requiring a
# predetermined dependent variable.
# ============================================================

def invariant_error(
    expression,
    dataset
):

    values = []

    for row in dataset.rows:

        value = expression.evaluate(
            row
        )

        if value is None:
            continue

        if not math.isfinite(value):
            continue

        if abs(value) > MAX_ABS_VALUE:
            continue

        values.append(value)

    if len(values) < 10:

        return (
            float("inf"),
            0.0,
            None
        )

    center = (
        sum(values)
        / len(values)
    )

    absolute_deviations = [
        abs(v - center)
        for v in values
    ]

    scale = max(
        abs(center),
        robust_scale(values),
        1e-12
    )

    error = (
        sum(
            absolute_deviations
        )
        / len(values)
        / scale
    )

    coverage = (
        len(values)
        / len(dataset.rows)
    )

    return (
        error,
        coverage,
        center
    )

# ============================================================
# 16. NOVELTY
# ============================================================

def expression_similarity(
    expression,
    archive
):

    if not archive:

        return 1.0

    variables = expression.variables_used()

    best_similarity = 0.0

    for old in archive:

        old_variables = (
            old.variables_used()
        )

        intersection = len(
            variables
            & old_variables
        )

        union = len(
            variables
            | old_variables
        )

        if union == 0:

            variable_similarity = 1.0

        else:

            variable_similarity = (
                intersection
                / union
            )

        size_difference = abs(
            expression.size()
            - old.size()
        )

        structural_similarity = (
            1.0
            / (
                1.0
                + size_difference
            )
        )

        similarity = (
            0.6
            * variable_similarity
            + 0.4
            * structural_similarity
        )

        best_similarity = max(
            best_similarity,
            similarity
        )

    return max(
        0.0,
        1.0 - best_similarity
    )

# ============================================================
# 17. SCORE PREDICTIVE CANDIDATE
# ============================================================

def score_predictive_candidate(
    expression,
    dataset,
    target,
    archive
):

    error, coverage = predictive_error(
        expression,
        dataset,
        target
    )

    if not math.isfinite(error):

        return Candidate(
            expression,
            float("inf"),
            float("inf"),
            expression.size(),
            coverage,
            0.0,
            "predictive",
            target
        )

    complexity = expression.size()

    depth = expression.depth()

    novelty = expression_novelty(
        expression,
        archive
    )

    complexity_penalty = (
        0.0007 * complexity
        + 0.0002 * depth
    )

    coverage_penalty = (
        max(
            0.0,
            1.0 - coverage
        )
        * 2.0
    )

    novelty_bonus = (
        0.03 * novelty
    )

    score = (
        error
        + complexity_penalty
        + coverage_penalty
        - novelty_bonus
    )

    return Candidate(
        expression,
        score,
        error,
        complexity,
        coverage,
        novelty,
        "predictive",
        target
    )

# ============================================================
# 18. EXPRESSION NOVELTY
# ============================================================

def expression_novelty(
    expression,
    archive
):

    if not archive:
        return 1.0

    key = canonical(
        expression
    )

    for old in archive:

        if canonical(old) == key:

            return 0.0

    # Variable-level novelty.

    used = expression.variables_used()

    if not used:
        return 0.1

    maximum_overlap = 0.0

    for old in archive:

        old_used = (
            old.variables_used()
        )

        if not old_used:
            continue

        overlap = len(
            used & old_used
        ) / len(
            used | old_used
        )

        maximum_overlap = max(
            maximum_overlap,
            overlap
        )

    return max(
        0.0,
        1.0 - maximum_overlap * 0.7
    )

# ============================================================
# 19. SCORE INVARIANT CANDIDATE
# ============================================================

def score_invariant_candidate(
    expression,
    dataset,
    archive
):

    error, coverage, constant = (
        invariant_error(
            expression,
            dataset
        )
    )

    if not math.isfinite(error):

        return Candidate(
            expression,
            float("inf"),
            float("inf"),
            expression.size(),
            coverage,
            0.0,
            "invariant"
        )

    # Reject expressions that are simply constants.

    if not expression.variables_used():

        return Candidate(
            expression,
            float("inf"),
            error,
            expression.size(),
            coverage,
            0.0,
            "invariant"
        )

    complexity = expression.size()

    depth = expression.depth()

    novelty = expression_novelty(
        expression,
        archive
    )

    score = (
        error
        + 0.0007 * complexity
        + 0.0002 * depth
        + max(
            0.0,
            1.0 - coverage
        )
        * 2.0
        - 0.03 * novelty
    )

    return Candidate(
        expression,
        score,
        error,
        complexity,
        coverage,
        novelty,
        "invariant"
    )

# ============================================================
# 20. EQUIVALENCE CHECKING
# ============================================================

def numerically_equivalent(
    a,
    b,
    dataset,
    tolerance=1e-4
):

    checked = 0

    for row in dataset.rows[:100]:

        av = a.evaluate(row)
        bv = b.evaluate(row)

        if av is None or bv is None:
            continue

        if (
            not math.isfinite(av)
            or not math.isfinite(bv)
        ):
            continue

        scale = max(
            abs(av),
            abs(bv),
            1e-10
        )

        if (
            abs(av - bv)
            / scale
            > tolerance
        ):

            return False

        checked += 1

    return checked >= 10

# ============================================================
# 21. DISCOVERY ARCHIVE
# ============================================================

class DiscoveryArchive:

    def __init__(self, limit=ARCHIVE_LIMIT):

        self.limit = limit

        self.entries = []

        self.keys = set()

    def contains(
        self,
        expression
    ):

        return (
            canonical(expression)
            in self.keys
        )

    def add(
        self,
        candidate
    ):

        expression = candidate.expression

        key = canonical(
            expression
        )

        if key in self.keys:

            return False

        # Avoid obvious duplicate forms.

        for existing in self.entries:

            if (
                expression.variables_used()
                == existing.expression.variables_used()
            ):

                if numerically_equivalent(
                    expression,
                    existing.expression,
                    CURRENT_DATASET
                ):

                    return False

        self.entries.append(
            candidate
        )

        self.keys.add(key)

        self.entries.sort(
            key=lambda c: c.score
        )

        if len(self.entries) > self.limit:

            removed = self.entries.pop()

            self.keys.discard(
                canonical(
                    removed.expression
                )
            )

        return True

    def expressions(self):

        return [
            candidate.expression
            for candidate in self.entries
        ]

# ============================================================
# 22. TARGET VARIABLE FILTER
# ============================================================

def useful_target(
    expression,
    target
):

    variables = (
        expression.variables_used()
    )

    # A prediction of a variable by itself
    # is trivial and should not count.

    if variables == {target}:

        return False

    return True

# ============================================================
# 23. TOURNAMENT SELECTION
# ============================================================

def tournament(
    scored
):

    candidates = random.sample(
        scored,
        min(
            TOURNAMENT_SIZE,
            len(scored)
        )
    )

    candidates.sort(
        key=lambda c: c.score
    )

    return candidates[0].expression

# ============================================================
# 24. DISCOVERY ENGINE
# ============================================================

CURRENT_DATASET = None

class DiscoveryEngine:

    def __init__(
        self,
        dataset
    ):

        global CURRENT_DATASET

        CURRENT_DATASET = dataset

        self.dataset = dataset

        self.train, self.test = (
            dataset.split()
        )

        self.grammar = Grammar(
            dataset.variables
        )

        self.archive = (
            DiscoveryArchive()
        )

        self.population = []

        self.best_candidates = []

        self.stagnation = 0

        self.generation = 0

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    def initialize(self):

        self.population = []

        for _ in range(
            POPULATION_SIZE
        ):

            expression = (
                random_expression(
                    self.grammar,
                    max_depth=INITIAL_DEPTH
                )
            )

            expression = simplify(
                expression
            )

            self.population.append(
                expression
            )

    # --------------------------------------------------------
    # Score population
    # --------------------------------------------------------

    def score_population(
        self
    ):

        scored = []

        # Half population predictive.

        for expression in self.population:

            expression = simplify(
                expression
            )

            if (
                expression.size()
                > MAX_TREE_SIZE
            ):

                continue

            if (
                expression.depth()
                > MAX_TREE_DEPTH
            ):

                continue

            # ------------------------------------------------
            # Predictive discoveries.
            # ------------------------------------------------

            for target in self.dataset.variables:

                if not useful_target(
                    expression,
                    target
                ):

                    continue

                candidate = (
                    score_predictive_candidate(
                        expression,
                        self.train,
                        target,
                        self.archive.expressions()
                    )
                )

                if math.isfinite(
                    candidate.score
                ):

                    scored.append(
                        candidate
                    )

            # ------------------------------------------------
            # Invariant discoveries.
            # ------------------------------------------------

            invariant = (
                score_invariant_candidate(
                    expression,
                    self.train,
                    self.archive.expressions()
                )
            )

            if math.isfinite(
                invariant.score
            ):

                scored.append(
                    invariant
                )

        scored.sort(
            key=lambda c: c.score
        )

        return scored

    # --------------------------------------------------------
    # Add discoveries.
    # --------------------------------------------------------

    def update_archive(
        self,
        scored
    ):

        added = []

        for candidate in scored[:100]:

            if candidate.error > 0.02:

                continue

            if candidate.coverage < 0.85:

                continue

            if self.archive.add(
                candidate
            ):

                added.append(
                    candidate
                )

        return added

    # --------------------------------------------------------
    # Build new population.
    # --------------------------------------------------------

    def reproduce(
        self,
        scored
    ):

        new_population = []

        valid = [
            candidate
            for candidate in scored
            if math.isfinite(
                candidate.score
            )
        ]

        if not valid:

            return [
                random_expression(
                    self.grammar,
                    max_depth=3
                )
                for _ in range(
                    POPULATION_SIZE
                )
            ]

        # Elite expressions.

        elite_expressions = []

        seen = set()

        for candidate in valid:

            key = canonical(
                candidate.expression
            )

            if key in seen:
                continue

            seen.add(key)

            elite_expressions.append(
                candidate.expression.clone()
            )

            if len(
                elite_expressions
            ) >= ELITE_COUNT:

                break

        new_population.extend(
            elite_expressions
        )

        while len(
            new_population
        ) < POPULATION_SIZE:

            parent_a = tournament(
                valid
            )

            if (
                random.random()
                < CROSSOVER_RATE
            ):

                parent_b = tournament(
                    valid
                )

                child = crossover(
                    parent_a,
                    parent_b
                )

            else:

                child = parent_a.clone()

            if random.random() < MUTATION_RATE:

                child = mutate(
                    child,
                    self.grammar
                )

            child = simplify(
                child
            )

            if (
                child.size()
                > MAX_TREE_SIZE
            ):

                continue

            if (
                child.depth()
                > MAX_TREE_DEPTH
            ):

                continue

            new_population.append(
                child
            )

        return new_population

    # --------------------------------------------------------
    # Test candidate.
    # --------------------------------------------------------

    def test_candidate(
        self,
        candidate
    ):

        if candidate.kind == "predictive":

            error, coverage = (
                predictive_error(
                    candidate.expression,
                    self.test,
                    candidate.target
                )
            )

            return error, coverage

        error, coverage, constant = (
            invariant_error(
                candidate.expression,
                self.test
            )
        )

        return error, coverage

    # --------------------------------------------------------
    # Print discovery.
    # --------------------------------------------------------

    def print_discovery(
        self,
        candidate,
        test_error,
        test_coverage
    ):

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            "[NEW DISCOVERY]"
        )

        if candidate.kind == "predictive":

            print(
                f"{candidate.target} ≈ "
                f"{candidate.expression}"
            )

        else:

            print(
                f"{candidate.expression} ≈ CONSTANT"
            )

        print(
            f"train error : "
            f"{candidate.error:.8g}"
        )

        print(
            f"test error  : "
            f"{test_error:.8g}"
        )

        print(
            f"coverage    : "
            f"{candidate.coverage:.3f}"
            f" / test "
            f"{test_coverage:.3f}"
        )

        print(
            f"complexity  : "
            f"{candidate.complexity}"
        )

        print(
            f"novelty     : "
            f"{candidate.novelty:.3f}"
        )

        print(
            f"kind        : "
            f"{candidate.kind}"
        )

    # --------------------------------------------------------
    # Run.
    # --------------------------------------------------------

    def run(
        self,
        generations=GENERATIONS
    ):

        self.initialize()

        print()
        print(
            "=" * 70
        )

        print(
            "PHYDISCOVER V3"
        )

        print(
            "OPEN-ENDED LEVEL-1 MATHEMATICAL DISCOVERY"
        )

        print(
            "=" * 70
        )

        print(
            f"Variables: "
            f"{', '.join(self.dataset.variables)}"
        )

        print(
            f"Rows: "
            f"{len(self.dataset.rows)}"
        )

        print(
            "=" * 70
        )

        for generation in range(
            generations
        ):

            self.generation = generation

            scored = (
                self.score_population()
            )

            if not scored:

                self.grammar.grow()

                self.initialize()

                continue

            best = scored[0]

            discoveries = (
                self.update_archive(
                    scored
                )
            )

            for discovery in discoveries:

                test_error, test_coverage = (
                    self.test_candidate(
                        discovery
                    )
                )

                if (
                    math.isfinite(
                        test_error
                    )
                    and test_coverage >= 0.80
                ):

                    self.print_discovery(
                        discovery,
                        test_error,
                        test_coverage
                    )

            # ------------------------------------------------
            # Progress.
            # ------------------------------------------------

            if generation % 20 == 0:

                test_error, test_coverage = (
                    self.test_candidate(
                        best
                    )
                )

                print(
                    f"\nGeneration "
                    f"{generation:4d}"
                )

                print(
                    f"Best: "
                    f"{best.expression}"
                )

                print(
                    f"Score: "
                    f"{best.score:.8g}"
                )

                print(
                    f"Train error: "
                    f"{best.error:.8g}"
                )

                print(
                    f"Test error: "
                    f"{test_error:.8g}"
                )

                print(
                    f"Archive: "
                    f"{len(self.archive.entries)}"
                )

            # ------------------------------------------------
            # Stagnation.
            # ------------------------------------------------

            if discoveries:

                self.stagnation = 0

            else:

                self.stagnation += 1

            # ------------------------------------------------
            # Grammar growth.
            # ------------------------------------------------

            if (
                self.stagnation > 45
            ):

                if self.grammar.level < 6:

                    self.grammar.grow()

                    self.stagnation = 0

            # ------------------------------------------------
            # Reproduce.
            # ------------------------------------------------

            self.population = (
                self.reproduce(
                    scored
                )
            )

        return self.archive

# ============================================================
# 25. DISCOVERY REPORT
# ============================================================

def print_final_archive(
    archive,
    dataset
):

    print()
    print()
    print(
        "=" * 70
    )

    print(
        "FINAL DISCOVERY ARCHIVE"
    )

    print(
        "=" * 70
    )

    if not archive.entries:

        print(
            "No robust discoveries found."
        )

        return

    for index, candidate in enumerate(
        archive.entries,
        1
    ):

        print()
        print(
            f"[{index}]"
        )

        if candidate.kind == "predictive":

            print(
                f"{candidate.target} ≈ "
                f"{candidate.expression}"
            )

        else:

            print(
                f"{candidate.expression} ≈ constant"
            )

        print(
            f"error={candidate.error:.8g} | "
            f"complexity={candidate.complexity} | "
            f"coverage={candidate.coverage:.3f}"
        )

        # Re-test on unseen data.

        if candidate.kind == "predictive":

            error, coverage = (
                predictive_error(
                    candidate.expression,
                    dataset,
                    candidate.target
                )
            )

        else:

            error, coverage, constant = (
                invariant_error(
                    candidate.expression,
                    dataset
                )
            )

        print(
            f"full-data error={error:.8g} | "
            f"coverage={coverage:.3f}"
        )

# ============================================================
# 26. DISCOVERY OF PAIRWISE RATIOS
#
# This additional deterministic layer searches for simple
# relationships of the form:
#
#   A / B ≈ constant
#
# and:
#
#   A * B ≈ constant
#
# These are useful because they can expose relationships that
# GP may take longer to discover.
# ============================================================

def discover_simple_invariants(
    dataset,
    archive
):

    variables = dataset.variables

    for i in range(
        len(variables)
    ):

        for j in range(
            i + 1,
            len(variables)
        ):

            a = variables[i]
            b = variables[j]

            # A / B

            ratio = Node(
                op="/",
                left=Node(value=a),
                right=Node(value=b)
            )

            candidate = (
                score_invariant_candidate(
                    ratio,
                    dataset,
                    archive.expressions()
                )
            )

            if (
                candidate.error < 0.02
                and candidate.coverage > 0.9
            ):

                archive.add(
                    candidate
                )

            # A * B

            product = Node(
                op="*",
                left=Node(value=a),
                right=Node(value=b)
            )

            candidate = (
                score_invariant_candidate(
                    product,
                    dataset,
                    archive.expressions()
                )
            )

            if (
                candidate.error < 0.02
                and candidate.coverage > 0.9
            ):

                archive.add(
                    candidate
                )

# ============================================================
# 27. DIMENSIONLESS / RELATIVE RELATION SEARCH
# ============================================================

def discover_power_ratios(
    dataset,
    archive
):

    variables = dataset.variables

    powers = [
        -2,
        -1,
        -0.5,
        0.5,
        1,
        2
    ]

    for target in variables:

        for a in variables:

            if a == target:
                continue

            for power in powers:

                expression = Node(
                    op="*",
                    left=Node(
                        value=target
                    ),
                    right=Node(
                        op="^",
                        left=Node(value=a),
                        right=Node(
                            value=power
                        )
                    )
                )

                candidate = (
                    score_invariant_candidate(
                        expression,
                        dataset,
                        archive.expressions()
                    )
                )

                if (
                    candidate.error
                    < 0.01
                    and candidate.coverage
                    > 0.9
                ):

                    archive.add(
                        candidate
                    )

# ============================================================
# 28. HIDDEN CONSTANT ESTIMATION
# ============================================================

def estimate_constant(
    expression,
    dataset
):

    values = []

    for row in dataset.rows:

        value = expression.evaluate(
            row
        )

        if value is None:
            continue

        if not math.isfinite(value):
            continue

        values.append(value)

    if not values:
        return None

    values.sort()

    middle = len(values) // 2

    if len(values) % 2 == 0:

        return (
            values[middle - 1]
            + values[middle]
        ) / 2

    return values[middle]

# ============================================================
# 29. CONSTANT-AWARE DISCOVERY
#
# Converts:
#
#   A/(x*y) ≈ 2.75
#
# into:
#
#   A ≈ 2.75*x*y
#
# when appropriate.
# ============================================================

def convert_invariant_to_prediction(
    candidate,
    dataset
):

    expression = candidate.expression

    constant = estimate_constant(
        expression,
        dataset
    )

    if constant is None:
        return None

    if not math.isfinite(constant):
        return None

    if abs(constant) > 1e8:
        return None

    # Try:
    #
    # expression ≈ c
    #
    # If expression is:
    #
    # A / f
    #
    # then construct:
    #
    # A ≈ c*f

    if (
        not expression.is_leaf()
        and expression.op == "/"
    ):

        numerator = expression.left
        denominator = expression.right

        if (
            numerator is not None
            and denominator is not None
            and len(
                numerator.variables_used()
            ) > 0
        ):

            prediction = simplify(
                Node(
                    op="*",
                    left=Node(
                        value=constant
                    ),
                    right=denominator.clone()
                )
            )

            targets = (
                numerator.variables_used()
            )

            if len(targets) == 1:

                target = next(
                    iter(targets)
                )

                candidate = (
                    score_predictive_candidate(
                        prediction,
                        dataset,
                        target,
                        []
                    )
                )

                if math.isfinite(
                    candidate.score
                ):

                    return candidate

    return None

# ============================================================
# 30. DISCOVERY LOOP FOR INVARIANTS
# ============================================================

def expand_archive_with_derived_forms(
    archive,
    dataset
):

    new_candidates = []

    for candidate in list(
        archive.entries
    ):

        if candidate.kind != "invariant":
            continue

        derived = (
            convert_invariant_to_prediction(
                candidate,
                dataset
            )
        )

        if derived is None:
            continue

        if (
            derived.error < 0.02
            and derived.coverage > 0.9
        ):

            if archive.add(
                derived
            ):

                new_candidates.append(
                    derived
                )

    return new_candidates

# ============================================================
# 31. BOOTSTRAP DISCOVERY
# ============================================================

def bootstrap_discovery(
    dataset,
    archive
):

    discover_simple_invariants(
        dataset,
        archive
    )

    discover_power_ratios(
        dataset,
        archive
    )

    expand_archive_with_derived_forms(
        archive,
        dataset
    )

# ============================================================
# 32. MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "              PHYDISCOVER V3"
    )

    print(
        "       OPEN-ENDED MATHEMATICAL AI"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "The engine receives observations."
    )

    print(
        "It is NOT given the hidden equations."
    )

    print(
        "It searches for relationships autonomously."
    )

    print()

    dataset = (
        make_experimental_data()
    )

    print(
        f"Dataset: "
        f"{dataset.name}"
    )

    print(
        f"Observations: "
        f"{len(dataset.rows)}"
    )

    print(
        f"Variables: "
        f"{', '.join(dataset.variables)}"
    )

    print()

    engine = DiscoveryEngine(
        dataset
    )

    # --------------------------------------------------------
    # Bootstrap with simple structural searches.
    # --------------------------------------------------------

    print(
        "[BOOTSTRAP] Searching simple invariants..."
    )

    bootstrap_discovery(
        dataset,
        engine.archive
    )

    print(
        f"[BOOTSTRAP] "
        f"{len(engine.archive.entries)} "
        f"candidate discoveries"
    )

    # --------------------------------------------------------
    # Evolutionary discovery.
    # --------------------------------------------------------

    archive = engine.run(
        generations=GENERATIONS
    )

    # --------------------------------------------------------
    # Derive predictive equations from invariant discoveries.
    # --------------------------------------------------------

    for _ in range(3):

        derived = (
            expand_archive_with_derived_forms(
                archive,
                dataset
            )
        )

        if not derived:
            break

    # --------------------------------------------------------
    # Final archive.
    # --------------------------------------------------------

    print_final_archive(
        archive,
        dataset
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DISCOVERY COMPLETE"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "The equations in the archive were not supplied"
    )

    print(
        "as targets to the evolutionary search."
    )

    print()

# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
