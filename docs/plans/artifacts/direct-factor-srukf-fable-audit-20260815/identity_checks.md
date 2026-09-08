=== I1: transpose(R) R = A transpose(A) given A transpose = Q R, Q orthonormal ===
{
  "status": "missing_assumptions",
  "reason": "The target has missing route-required assumptions.",
  "givens": [
    "A transpose = Q R"
  ],
  "target": "transpose(R) R = A transpose(A)",
  "lhs": "transpose(R) R",
  "rhs": "A transpose(A)",
  "route_decision": {
    "route": "human_review",
    "status": "unknown",
    "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
    "backend_attempt": {
      "backend": "router",
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "evidence": [],
      "severity": "diagnostic"
    },
    "obligation": {
      "id": "route-obligation-1",
      "lhs": "transpose(R) R",
      "rhs": "A transpose(A)",
      "assumptions": [],
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexample": null,
      "missing_assumptions": [],
      "provenance": {}
    },
    "result": {
      "question": {
        "question_type": "route_obligation",
        "target": "transpose(R) R == A transpose(A)",
        "givens": [],
        "assumptions": [],
        "context": {
          "route": "human_review"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "obligations": [
        {
          "id": "route-obligation-1",
          "lhs": "transpose(R) R",
          "rhs": "A transpose(A)",
          "assumptions": [],
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "backend_attempts": [
            {
              "backend": "router",
              "status": "unknown",
              "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [
        {
          "kind": "review_or_reencode",
          "route": "human_review"
        }
      ],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_route_decision"
    }
  },
  "assumption_diagnostic": {
    "status": "missing_assumptions",
    "reason": "At least one route-required assumption is missing.",
    "target": "transpose(R) R = A transpose(A)",
    "provided_assumptions": [
      "Q transpose Q = I"
    ],
    "assumptions": [
      {
        "text": "matrix dimensions are conformable for the operation",
        "status": "missing",
        "source": "matrix operation",
        "necessity": "required_by_route",
        "used_by": [
          "matrix_conformability"
        ],
        "route_categories": [
          "shape_condition"
        ],
        "route_category_sources": [
          "assumption_rule:matrix_conformability"
        ]
      }
    ],
    "missing_assumptions": [
      {
        "text": "matrix dimensions are conformable for the operation",
        "status": "missing",
        "source": "matrix operation",
        "necessity": "required_by_route",
        "used_by": [
          "matrix_conformability"
        ],
        "route_categories": [
          "shape_condition"
        ],
        "route_category_sources": [
          "assumption_rule:matrix_conformability"
        ]
      }
    ],
    "workbench_result": {
      "question": {
        "question_type": "assumptions_required",
        "target": "transpose(R) R = A transpose(A)",
        "givens": [],
        "assumptions": [
          "Q transpose Q = I"
        ],
        "context": {
          "necessity_boundary": "route-required or sufficient, not minimal necessity"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "missing_assumptions",
      "reason": "At least one route-required assumption is missing.",
      "obligations": [
        {
          "id": "assumption-obligation-1",
          "lhs": "transpose(R) R = A transpose(A)",
          "rhs": "well-posed under route assumptions",
          "assumptions": [
            "Q transpose Q = I"
          ],
          "status": "missing_assumptions",
          "reason": "At least one route-required assumption is missing.",
          "backend_attempts": [],
          "counterexample": null,
          "missing_assumptions": [
            {
              "text": "matrix dimensions are conformable for the operation",
              "status": "missing",
              "source": "matrix operation",
              "necessity": "required_by_route",
              "used_by": [
                "matrix_conformability"
              ],
              "route_categories": [
                "shape_condition"
              ],
              "route_category_sources": [
                "assumption_rule:matrix_conformability"
              ]
            }
          ],
          "provenance": {}
        }
      ],
      "assumptions": [
        {
          "text": "matrix dimensions are conformable for the operation",
          "status": "missing",
          "source": "matrix operation",
          "necessity": "required_by_route",
          "used_by": [
            "matrix_conformability"
          ],
          "route_categories": [
            "shape_condition"
          ],
          "route_category_sources": [
            "assumption_rule:matrix_conformability"
          ]
        }
      ],
      "backend_attempts": [],
      "counterexamples": [],
      "actions": [
        {
          "kind": "state_or_verify_assumption",
          "assumption": "matrix dimensions are conformable for the operation",
          "source": "matrix operation"
        }
      ],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "assumption_discovery_result"
    }
  },
  "counterexample_search": {
    "status": "backend_unavailable",
    "reason": "SymPy is not installed in this environment.",
    "lhs": "transpose(R) R",
    "rhs": "A transpose(A)",
    "backend": "sympy_finite_domain",
    "search_space": {
      "kind": "finite_integer_domain",
      "domain": [
        -2,
        -1,
        0,
        1,
        2
      ],
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment."
    },
    "counterexample": null,
    "workbench_result": {
      "question": {
        "question_type": "find_counterexample",
        "target": "transpose(R) R == A transpose(A)",
        "givens": [],
        "assumptions": [],
        "context": {
          "domain": [
            -2,
            -1,
            0,
            1,
            2
          ]
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment.",
      "obligations": [
        {
          "id": "counterexample-obligation-1",
          "lhs": "transpose(R) R",
          "rhs": "A transpose(A)",
          "assumptions": [],
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "backend_attempts": [
            {
              "backend": "sympy_finite_domain",
              "status": "backend_unavailable",
              "reason": "SymPy is not installed in this environment.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "sympy_finite_domain",
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "counterexample_search_result"
    }
  },
  "workbench_result": {
    "question": {
      "question_type": "derive_or_refute",
      "target": "transpose(R) R = A transpose(A)",
      "givens": [
        "A transpose = Q R"
      ],
      "assumptions": [
        "Q transpose Q = I"
      ],
      "context": {
        "backend": "auto",
        "claim_semantics": {
          "status": "generic_theorem_route",
          "role": "theorem",
          "requested_authority": "none",
          "effective_authority": "implicit_generic_theorem",
          "routing_effect": "ordinary_proof_or_counterexample",
          "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
          "source": {},
          "non_claims": [
            "Claim-role evidence controls routing only; it is not a proof certificate.",
            "A source definition or identity is not thereby economically or scientifically valid."
          ],
          "metadata": {
            "schema_version": "1.0",
            "contract": "claim_semantics_validation"
          }
        }
      },
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_question"
      }
    },
    "status": "missing_assumptions",
    "reason": "The target has missing route-required assumptions.",
    "obligations": [
      {
        "id": "derive-target-1",
        "lhs": "transpose(R) R",
        "rhs": "A transpose(A)",
        "assumptions": [
          "Q transpose Q = I"
        ],
        "status": "missing_assumptions",
        "reason": "The target has missing route-required assumptions.",
        "backend_attempts": [
          {
            "backend": "router",
            "status": "unknown",
            "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
            "evidence": [],
            "severity": "diagnostic"
          }
        ],
        "counterexample": null,
        "missing_assumptions": [
          {
            "text": "matrix dimensions are conformable for the operation",
            "status": "missing",
            "source": "matrix operation",
            "necessity": "required_by_route",
            "used_by": [
              "matrix_conformability"
            ],
            "route_categories": [
              "shape_condition"
            ],
            "route_category_sources": [
              "assumption_rule:matrix_conformability"
            ]
          }
        ],
        "provenance": {}
      }
    ],
    "assumptions": [
      {
        "text": "matrix dimensions are conformable for the operation",
        "status": "missing",
        "source": "matrix operation",
        "necessity": "required_by_route",
        "used_by": [
          "matrix_conformability"
        ],
        "route_categories": [
          "shape_condition"
        ],
        "route_category_sources": [
          "assumption_rule:matrix_conformability"
        ]
      }
    ],
    "backend_attempts": [
      {
        "backend": "router",
        "status": "unknown",
        "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
        "evidence": [],
        "severity": "diagnostic"
      }
    ],
    "counterexamples": [],
    "actions": [
      {
        "kind": "state_or_verify_assumption",
        "assumption": "matrix dimensions are conformable for the operation",
        "source": "matrix operation"
      }
    ],
    "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_workbench_result"
    }
  },
  "claim_semantics": {
    "status": "generic_theorem_route",
    "role": "theorem",
    "requested_authority": "none",
    "effective_authority": "implicit_generic_theorem",
    "routing_effect": "ordinary_proof_or_counterexample",
    "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
    "source": {},
    "non_claims": [
      "Claim-role evidence controls routing only; it is not a proof certificate.",
      "A source definition or identity is not thereby economically or scientifically valid."
    ],
    "metadata": {
      "schema_version": "1.0",
      "contract": "claim_semantics_validation"
    }
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "derive_or_refute_result"
  }
}
=== I2: product rule d(S S^T) ===
{
  "status": "unknown",
  "reason": "No bounded derivation or refutation was found.",
  "givens": [],
  "target": "d(S S^T) = dS S^T + S dS^T",
  "lhs": "d(S S^T)",
  "rhs": "dS S^T + S dS^T",
  "route_decision": {
    "route": "human_review",
    "status": "unknown",
    "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
    "backend_attempt": {
      "backend": "router",
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "evidence": [],
      "severity": "diagnostic"
    },
    "obligation": {
      "id": "route-obligation-1",
      "lhs": "d(S S^T)",
      "rhs": "dS S^T + S dS^T",
      "assumptions": [],
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexample": null,
      "missing_assumptions": [],
      "provenance": {}
    },
    "result": {
      "question": {
        "question_type": "route_obligation",
        "target": "d(S S^T) == dS S^T + S dS^T",
        "givens": [],
        "assumptions": [],
        "context": {
          "route": "human_review"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "obligations": [
        {
          "id": "route-obligation-1",
          "lhs": "d(S S^T)",
          "rhs": "dS S^T + S dS^T",
          "assumptions": [],
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "backend_attempts": [
            {
              "backend": "router",
              "status": "unknown",
              "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [
        {
          "kind": "review_or_reencode",
          "route": "human_review"
        }
      ],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_route_decision"
    }
  },
  "assumption_diagnostic": {
    "status": "unknown",
    "reason": "No route-required assumptions were detected by the bounded rule set.",
    "target": "d(S S^T) = dS S^T + S dS^T",
    "provided_assumptions": [],
    "assumptions": [],
    "missing_assumptions": [],
    "workbench_result": {
      "question": {
        "question_type": "assumptions_required",
        "target": "d(S S^T) = dS S^T + S dS^T",
        "givens": [],
        "assumptions": [],
        "context": {
          "necessity_boundary": "route-required or sufficient, not minimal necessity"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "unknown",
      "reason": "No route-required assumptions were detected by the bounded rule set.",
      "obligations": [
        {
          "id": "assumption-obligation-1",
          "lhs": "d(S S^T) = dS S^T + S dS^T",
          "rhs": "well-posed under route assumptions",
          "assumptions": [],
          "status": "unknown",
          "reason": "No route-required assumptions were detected by the bounded rule set.",
          "backend_attempts": [],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [],
      "counterexamples": [],
      "actions": [],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "assumption_discovery_result"
    }
  },
  "counterexample_search": {
    "status": "backend_unavailable",
    "reason": "SymPy is not installed in this environment.",
    "lhs": "d(S S^T)",
    "rhs": "dS S^T + S dS^T",
    "backend": "sympy_finite_domain",
    "search_space": {
      "kind": "finite_integer_domain",
      "domain": [
        -2,
        -1,
        0,
        1,
        2
      ],
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment."
    },
    "counterexample": null,
    "workbench_result": {
      "question": {
        "question_type": "find_counterexample",
        "target": "d(S S^T) == dS S^T + S dS^T",
        "givens": [],
        "assumptions": [],
        "context": {
          "domain": [
            -2,
            -1,
            0,
            1,
            2
          ]
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment.",
      "obligations": [
        {
          "id": "counterexample-obligation-1",
          "lhs": "d(S S^T)",
          "rhs": "dS S^T + S dS^T",
          "assumptions": [],
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "backend_attempts": [
            {
              "backend": "sympy_finite_domain",
              "status": "backend_unavailable",
              "reason": "SymPy is not installed in this environment.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "sympy_finite_domain",
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "counterexample_search_result"
    }
  },
  "workbench_result": {
    "question": {
      "question_type": "derive_or_refute",
      "target": "d(S S^T) = dS S^T + S dS^T",
      "givens": [],
      "assumptions": [],
      "context": {
        "backend": "auto",
        "claim_semantics": {
          "status": "generic_theorem_route",
          "role": "theorem",
          "requested_authority": "none",
          "effective_authority": "implicit_generic_theorem",
          "routing_effect": "ordinary_proof_or_counterexample",
          "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
          "source": {},
          "non_claims": [
            "Claim-role evidence controls routing only; it is not a proof certificate.",
            "A source definition or identity is not thereby economically or scientifically valid."
          ],
          "metadata": {
            "schema_version": "1.0",
            "contract": "claim_semantics_validation"
          }
        }
      },
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_question"
      }
    },
    "status": "unknown",
    "reason": "No bounded derivation or refutation was found.",
    "obligations": [
      {
        "id": "derive-target-1",
        "lhs": "d(S S^T)",
        "rhs": "dS S^T + S dS^T",
        "assumptions": [],
        "status": "unknown",
        "reason": "No bounded derivation or refutation was found.",
        "backend_attempts": [
          {
            "backend": "router",
            "status": "unknown",
            "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
            "evidence": [],
            "severity": "diagnostic"
          }
        ],
        "counterexample": null,
        "missing_assumptions": [],
        "provenance": {}
      }
    ],
    "assumptions": [],
    "backend_attempts": [
      {
        "backend": "router",
        "status": "unknown",
        "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
        "evidence": [],
        "severity": "diagnostic"
      }
    ],
    "counterexamples": [],
    "actions": [
      {
        "kind": "manual_derivation_or_stronger_backend",
        "reason": "No bounded route resolved the target."
      }
    ],
    "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_workbench_result"
    }
  },
  "claim_semantics": {
    "status": "generic_theorem_route",
    "role": "theorem",
    "requested_authority": "none",
    "effective_authority": "implicit_generic_theorem",
    "routing_effect": "ordinary_proof_or_counterexample",
    "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
    "source": {},
    "non_claims": [
      "Claim-role evidence controls routing only; it is not a proof certificate.",
      "A source definition or identity is not thereby economically or scientifically valid."
    ],
    "metadata": {
      "schema_version": "1.0",
      "contract": "claim_semantics_validation"
    }
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "derive_or_refute_result"
  }
}
=== I3: d(z^T z) = 2 z^T dz ===
{
  "status": "unknown",
  "reason": "No bounded derivation or refutation was found.",
  "givens": [],
  "target": "d(z^T z) = 2 z^T dz",
  "lhs": "d(z^T z)",
  "rhs": "2 z^T dz",
  "route_decision": {
    "route": "human_review",
    "status": "unknown",
    "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
    "backend_attempt": {
      "backend": "router",
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "evidence": [],
      "severity": "diagnostic"
    },
    "obligation": {
      "id": "route-obligation-1",
      "lhs": "d(z^T z)",
      "rhs": "2 z^T dz",
      "assumptions": [],
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexample": null,
      "missing_assumptions": [],
      "provenance": {}
    },
    "result": {
      "question": {
        "question_type": "route_obligation",
        "target": "d(z^T z) == 2 z^T dz",
        "givens": [],
        "assumptions": [],
        "context": {
          "route": "human_review"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "unknown",
      "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
      "obligations": [
        {
          "id": "route-obligation-1",
          "lhs": "d(z^T z)",
          "rhs": "2 z^T dz",
          "assumptions": [],
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "backend_attempts": [
            {
              "backend": "router",
              "status": "unknown",
              "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "router",
          "status": "unknown",
          "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [
        {
          "kind": "review_or_reencode",
          "route": "human_review"
        }
      ],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_route_decision"
    }
  },
  "assumption_diagnostic": {
    "status": "unknown",
    "reason": "No route-required assumptions were detected by the bounded rule set.",
    "target": "d(z^T z) = 2 z^T dz",
    "provided_assumptions": [],
    "assumptions": [],
    "missing_assumptions": [],
    "workbench_result": {
      "question": {
        "question_type": "assumptions_required",
        "target": "d(z^T z) = 2 z^T dz",
        "givens": [],
        "assumptions": [],
        "context": {
          "necessity_boundary": "route-required or sufficient, not minimal necessity"
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "unknown",
      "reason": "No route-required assumptions were detected by the bounded rule set.",
      "obligations": [
        {
          "id": "assumption-obligation-1",
          "lhs": "d(z^T z) = 2 z^T dz",
          "rhs": "well-posed under route assumptions",
          "assumptions": [],
          "status": "unknown",
          "reason": "No route-required assumptions were detected by the bounded rule set.",
          "backend_attempts": [],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [],
      "counterexamples": [],
      "actions": [],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "assumption_discovery_result"
    }
  },
  "counterexample_search": {
    "status": "backend_unavailable",
    "reason": "SymPy is not installed in this environment.",
    "lhs": "d(z^T z)",
    "rhs": "2 z^T dz",
    "backend": "sympy_finite_domain",
    "search_space": {
      "kind": "finite_integer_domain",
      "domain": [
        -2,
        -1,
        0,
        1,
        2
      ],
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment."
    },
    "counterexample": null,
    "workbench_result": {
      "question": {
        "question_type": "find_counterexample",
        "target": "d(z^T z) == 2 z^T dz",
        "givens": [],
        "assumptions": [],
        "context": {
          "domain": [
            -2,
            -1,
            0,
            1,
            2
          ]
        },
        "metadata": {
          "schema_version": "1.0",
          "contract": "math_debugging_question"
        }
      },
      "status": "backend_unavailable",
      "reason": "SymPy is not installed in this environment.",
      "obligations": [
        {
          "id": "counterexample-obligation-1",
          "lhs": "d(z^T z)",
          "rhs": "2 z^T dz",
          "assumptions": [],
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "backend_attempts": [
            {
              "backend": "sympy_finite_domain",
              "status": "backend_unavailable",
              "reason": "SymPy is not installed in this environment.",
              "evidence": [],
              "severity": "diagnostic"
            }
          ],
          "counterexample": null,
          "missing_assumptions": [],
          "provenance": {}
        }
      ],
      "assumptions": [],
      "backend_attempts": [
        {
          "backend": "sympy_finite_domain",
          "status": "backend_unavailable",
          "reason": "SymPy is not installed in this environment.",
          "evidence": [],
          "severity": "diagnostic"
        }
      ],
      "counterexamples": [],
      "actions": [],
      "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_workbench_result"
      }
    },
    "metadata": {
      "schema_version": "1.0",
      "contract": "counterexample_search_result"
    }
  },
  "workbench_result": {
    "question": {
      "question_type": "derive_or_refute",
      "target": "d(z^T z) = 2 z^T dz",
      "givens": [],
      "assumptions": [],
      "context": {
        "backend": "auto",
        "claim_semantics": {
          "status": "generic_theorem_route",
          "role": "theorem",
          "requested_authority": "none",
          "effective_authority": "implicit_generic_theorem",
          "routing_effect": "ordinary_proof_or_counterexample",
          "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
          "source": {},
          "non_claims": [
            "Claim-role evidence controls routing only; it is not a proof certificate.",
            "A source definition or identity is not thereby economically or scientifically valid."
          ],
          "metadata": {
            "schema_version": "1.0",
            "contract": "claim_semantics_validation"
          }
        }
      },
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_question"
      }
    },
    "status": "unknown",
    "reason": "No bounded derivation or refutation was found.",
    "obligations": [
      {
        "id": "derive-target-1",
        "lhs": "d(z^T z)",
        "rhs": "2 z^T dz",
        "assumptions": [],
        "status": "unknown",
        "reason": "No bounded derivation or refutation was found.",
        "backend_attempts": [
          {
            "backend": "router",
            "status": "unknown",
            "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
            "evidence": [],
            "severity": "diagnostic"
          }
        ],
        "counterexample": null,
        "missing_assumptions": [],
        "provenance": {}
      }
    ],
    "assumptions": [],
    "backend_attempts": [
      {
        "backend": "router",
        "status": "unknown",
        "reason": "Expression appears to require matrix/domain review before scalar symbolic routing.",
        "evidence": [],
        "severity": "diagnostic"
      }
    ],
    "counterexamples": [],
    "actions": [
      {
        "kind": "manual_derivation_or_stronger_backend",
        "reason": "No bounded route resolved the target."
      }
    ],
    "certification_boundary": "Only deterministic backend certificates for scoped obligations can certify mathematical claims. Supporting, diagnostic, and numeric evidence must not be promoted to proof.",
    "metadata": {
      "schema_version": "1.0",
      "contract": "math_debugging_workbench_result"
    }
  },
  "claim_semantics": {
    "status": "generic_theorem_route",
    "role": "theorem",
    "requested_authority": "none",
    "effective_authority": "implicit_generic_theorem",
    "routing_effect": "ordinary_proof_or_counterexample",
    "reason": "No source-evidenced claim role was supplied; generic equality semantics apply.",
    "source": {},
    "non_claims": [
      "Claim-role evidence controls routing only; it is not a proof certificate.",
      "A source definition or identity is not thereby economically or scientifically valid."
    ],
    "metadata": {
      "schema_version": "1.0",
      "contract": "claim_semantics_validation"
    }
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "derive_or_refute_result"
  }
}
