#!/usr/bin/env python3
"""Exact retained-prefix squared-TT value and derivative certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias


Exponent: TypeAlias = tuple[int, ...]
Polynomial: TypeAlias = dict[Exponent, Fraction]
PolyMatrix: TypeAlias = list[list[Polynomial]]
Core: TypeAlias = list[list[list[Fraction]]]


def q(value: int, denominator: int = 1) -> Fraction:
    return Fraction(value, denominator)


def poly_constant(dimension: int, value: Fraction) -> Polynomial:
    return {(0,) * dimension: value} if value else {}


def poly_variable(dimension: int, axis: int) -> Polynomial:
    exponent = [0] * dimension
    exponent[axis] = 1
    return {tuple(exponent): q(1)}


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, q(0)) + coefficient
        if not result[exponent]:
            del result[exponent]
    return result


def poly_scale(polynomial: Polynomial, scalar: Fraction) -> Polynomial:
    if not scalar:
        return {}
    return {
        exponent: scalar * coefficient
        for exponent, coefficient in polynomial.items()
        if scalar * coefficient
    }


def poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            result[exponent] = (
                result.get(exponent, q(0)) + left_coefficient * right_coefficient
            )
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def matrix_add(left: PolyMatrix, right: PolyMatrix) -> PolyMatrix:
    return [
        [poly_add(left[row][column], right[row][column]) for column in range(len(left[0]))]
        for row in range(len(left))
    ]


def matrix_multiply(left: PolyMatrix, right: PolyMatrix) -> PolyMatrix:
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    if len(left[0]) != inner:
        raise ValueError("incompatible polynomial matrix shapes")
    dimension = len(next(iter(left[0][0]))) if left[0][0] else len(next(iter(right[0][0])))
    result = [[{} for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            total = poly_constant(dimension, q(0))
            for index in range(inner):
                total = poly_add(total, poly_multiply(left[row][index], right[index][column]))
            result[row][column] = total
    return result


def core_matrix(core: Core, dimension: int, axis: int) -> PolyMatrix:
    variable = poly_variable(dimension, axis)
    rows = len(core)
    columns = len(core[0][0])
    result: PolyMatrix = [[{} for _ in range(columns)] for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            value = poly_constant(dimension, core[row][0][column])
            linear = poly_scale(variable, core[row][1][column])
            result[row][column] = poly_add(value, linear)
    return result


def tt_and_directional_derivative(
    cores: list[Core], dotted_cores: list[Core], dimension: int
) -> tuple[Polynomial, Polynomial]:
    value: PolyMatrix = [[poly_constant(dimension, q(1))]]
    derivative: PolyMatrix = [[poly_constant(dimension, q(0))]]
    for axis, (core, dotted_core) in enumerate(zip(cores, dotted_cores)):
        factor = core_matrix(core, dimension, axis)
        dotted_factor = core_matrix(dotted_core, dimension, axis)
        derivative = matrix_add(
            matrix_multiply(derivative, factor),
            matrix_multiply(value, dotted_factor),
        )
        value = matrix_multiply(value, factor)
    return value[0][0], derivative[0][0]


def uniform_moment(power: int) -> Fraction:
    if power % 2:
        return q(0)
    return q(1, power + 1)


def integrate_trailing(polynomial: Polynomial, retained: int) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        integrated_coefficient = coefficient
        for power in exponent[retained:]:
            integrated_coefficient *= uniform_moment(power)
        retained_exponent = exponent[:retained]
        result[retained_exponent] = (
            result.get(retained_exponent, q(0)) + integrated_coefficient
        )
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def numeric_mass_contraction(
    cores: list[Core], dotted_cores: list[Core], retained: int
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    mass = [[q(1)]]
    dotted_mass = [[q(0)]]
    basis_mass = [[q(1), q(0)], [q(0), q(1, 3)]]
    for axis in range(len(cores) - 1, retained - 1, -1):
        core = cores[axis]
        dotted_core = dotted_cores[axis]
        left_rank = len(core)
        right_rank = len(core[0][0])
        next_mass = [[q(0) for _ in range(left_rank)] for _ in range(left_rank)]
        next_dotted = [[q(0) for _ in range(left_rank)] for _ in range(left_rank)]
        for left in range(left_rank):
            for left_prime in range(left_rank):
                for basis in range(2):
                    for basis_prime in range(2):
                        basis_weight = basis_mass[basis][basis_prime]
                        for right in range(right_rank):
                            for right_prime in range(right_rank):
                                weight = basis_weight * mass[right][right_prime]
                                dotted_weight = basis_weight * dotted_mass[right][right_prime]
                                next_mass[left][left_prime] += (
                                    core[left][basis][right]
                                    * core[left_prime][basis_prime][right_prime]
                                    * weight
                                )
                                next_dotted[left][left_prime] += (
                                    dotted_core[left][basis][right]
                                    * core[left_prime][basis_prime][right_prime]
                                    * weight
                                    + core[left][basis][right]
                                    * dotted_core[left_prime][basis_prime][right_prime]
                                    * weight
                                    + core[left][basis][right]
                                    * core[left_prime][basis_prime][right_prime]
                                    * dotted_weight
                                )
        mass, dotted_mass = next_mass, next_dotted
    return mass, dotted_mass


def retained_prefix(
    cores: list[Core], dotted_cores: list[Core], retained: int
) -> tuple[PolyMatrix, PolyMatrix]:
    value: PolyMatrix = [[poly_constant(retained, q(1))]]
    derivative: PolyMatrix = [[poly_constant(retained, q(0))]]
    for axis in range(retained):
        factor = core_matrix(cores[axis], retained, axis)
        dotted_factor = core_matrix(dotted_cores[axis], retained, axis)
        derivative = matrix_add(
            matrix_multiply(derivative, factor),
            matrix_multiply(value, dotted_factor),
        )
        value = matrix_multiply(value, factor)
    return value, derivative


def quadratic_form(
    left: PolyMatrix, middle: list[list[Fraction]], right: PolyMatrix
) -> Polynomial:
    dimension = len(next(iter(left[0][0])))
    total = poly_constant(dimension, q(0))
    for index in range(len(middle)):
        for other in range(len(middle[0])):
            term = poly_multiply(left[0][index], right[0][other])
            total = poly_add(total, poly_scale(term, middle[index][other]))
    return total


def contraction_value_and_derivative(
    cores: list[Core], dotted_cores: list[Core], retained: int
) -> tuple[Polynomial, Polynomial]:
    prefix, dotted_prefix = retained_prefix(cores, dotted_cores, retained)
    mass, dotted_mass = numeric_mass_contraction(cores, dotted_cores, retained)
    value = quadratic_form(prefix, mass, prefix)
    derivative = poly_add(
        poly_add(
            quadratic_form(dotted_prefix, mass, prefix),
            quadratic_form(prefix, dotted_mass, prefix),
        ),
        quadratic_form(prefix, mass, dotted_prefix),
    )
    return value, derivative


def apply_fixed_scale_and_defensive_mass(
    polynomial: Polynomial, scale: Fraction, tau: Fraction, retained: int
) -> Polynomial:
    # The defensive density is one under the normalized uniform reference measure.
    return poly_add(
        poly_scale(polynomial, scale),
        poly_constant(retained, scale * tau),
    )


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def polynomial_record(polynomial: Polynomial) -> dict[str, str]:
    return {
        ",".join(str(power) for power in exponent): fraction_text(coefficient)
        for exponent, coefficient in sorted(polynomial.items())
    }


def case_record(
    name: str,
    cores: list[Core],
    dotted_cores: list[Core],
    retained: int,
    scale: Fraction,
    tau: Fraction,
) -> dict[str, object]:
    dimension = len(cores)
    phi, dotted_phi = tt_and_directional_derivative(cores, dotted_cores, dimension)
    direct_square = integrate_trailing(poly_multiply(phi, phi), retained)
    direct_derivative = integrate_trailing(
        poly_scale(poly_multiply(phi, dotted_phi), q(2)), retained
    )
    contraction_square, contraction_derivative = contraction_value_and_derivative(
        cores, dotted_cores, retained
    )
    direct_value = apply_fixed_scale_and_defensive_mass(
        direct_square, scale, tau, retained
    )
    contraction_value = apply_fixed_scale_and_defensive_mass(
        contraction_square, scale, tau, retained
    )
    direct_dot = poly_scale(direct_derivative, scale)
    contraction_dot = poly_scale(contraction_derivative, scale)
    value_equal = direct_value == contraction_value
    derivative_equal = direct_dot == contraction_dot
    return {
        "name": name,
        "dimension": dimension,
        "retained_dimension": retained,
        "integrated_dimension": dimension - retained,
        "ranks": [len(cores[0])] + [len(core[0][0]) for core in cores],
        "value_equal": value_equal,
        "derivative_equal": derivative_equal,
        "direct_value": polynomial_record(direct_value),
        "contraction_value": polynomial_record(contraction_value),
        "direct_directional_derivative": polynomial_record(direct_dot),
        "contraction_directional_derivative": polynomial_record(contraction_dot),
    }


def scalar_case() -> tuple[list[Core], list[Core]]:
    cores = [
        [[[q(1), q(2)], [q(1), q(-1)]]],
        [[[q(1)], [q(2)]], [[q(-1)], [q(1)]]],
    ]
    dotted = [
        [[[q(1), q(-1)], [q(0), q(2)]]],
        [[[q(0)], [q(1)]], [[q(2)], [q(-1)]]],
    ]
    return cores, dotted


def vector_case() -> tuple[list[Core], list[Core]]:
    cores = [
        [[[q(1), q(2)], [q(1), q(-1)]]],
        [
            [[q(1), q(0)], [q(1), q(-1)]],
            [[q(0), q(1)], [q(2), q(1)]],
        ],
        [
            [[q(1), q(1)], [q(0), q(1)]],
            [[q(-1), q(2)], [q(1), q(0)]],
        ],
        [[[q(1)], [q(1)]], [[q(2)], [q(-1)]]],
    ]
    dotted = [
        [[[q(0), q(1)], [q(1), q(0)]]],
        [
            [[q(1), q(-1)], [q(0), q(1)]],
            [[q(0), q(1)], [q(-1), q(0)]],
        ],
        [
            [[q(0), q(1)], [q(1), q(-1)]],
            [[q(1), q(0)], [q(0), q(1)]],
        ],
        [[[q(1)], [q(0)]], [[q(-1)], [q(2)]]],
    ]
    return cores, dotted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    scale = q(2, 3)
    tau = q(1, 5)
    cases = []
    for name, retained, factory in (
        ("scalar_retained_prefix_m1_d2", 1, scalar_case),
        ("vector_retained_prefix_m2_d4", 2, vector_case),
    ):
        cores, dotted_cores = factory()
        cases.append(case_record(name, cores, dotted_cores, retained, scale, tau))
    passed = all(
        bool(case["value_equal"]) and bool(case["derivative_equal"])
        for case in cases
    )
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "schema": "fable_rewrite_squared_tt_retained_block_certificate.v1",
        "status": "pass" if passed else "fail",
        "arithmetic": "exact_python_fraction",
        "reference_measure": "product_normalized_uniform_on_minus1_1",
        "basis": ["1", "z"],
        "basis_mass": [["1", "0"], ["0", "1/3"]],
        "fixed_scale_exp_minus_c": fraction_text(scale),
        "fixed_tau": fraction_text(tau),
        "differentiated_objects": ["tt_cores"],
        "frozen_objects": [
            "basis",
            "basis_mass",
            "reference_measure",
            "domain",
            "coordinate_map",
            "c",
            "tau",
            "defensive_density",
        ],
        "source_sha256": source_digest,
        "cases": cases,
        "nonclaims": [
            "does_not_certify_tt_fitting",
            "does_not_certify_adaptive_branch_selection",
            "does_not_certify_runtime_implementation_parity",
            "does_not_certify_hmc_or_posterior_validity",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": cases}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
