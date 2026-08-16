=== Q1: S S^T = A A^T for S = transpose(R), A^T = Q R ===
{
  "status": "missing_assumptions",
  "workflow": "assumptions_for",
  "question": "What assumptions are required for S S^T = A A^T for S = transpose(R), A^T = Q R?",
  "claim_class": "assumption_discovery",
  "answer": "At least one route-required assumption is missing.",
  "evidence": [
    {
      "id": "assumptions_for:missing-assumption",
      "class": "missing_assumption",
      "source": "route",
      "summary": "At least one route-required assumption is missing.",
      "low_level": {
        "status": "missing_assumptions",
        "reason": "At least one route-required assumption is missing.",
        "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
        "provided_assumptions": [],
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
            "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
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
          "status": "missing_assumptions",
          "reason": "At least one route-required assumption is missing.",
          "obligations": [
            {
              "id": "assumption-obligation-1",
              "lhs": "S S^T = A A^T for S = transpose(R), A^T = Q R",
              "rhs": "well-posed under route assumptions",
              "assumptions": [],
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
      }
    }
  ],
  "evidence_classes": [
    "missing_assumption"
  ],
  "certification_source": "none",
  "veto_reasons": [],
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
  "counterexamples": [],
  "actions": [
    {
      "code": "human_review",
      "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
    },
    {
      "code": "review_assumption_proposals",
      "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
    }
  ],
  "non_claims": [
    {
      "code": "general_theorem_proving_not_claimed",
      "text": "This scoped workflow result does not claim general theorem-proving ability."
    },
    {
      "code": "release_readiness_not_claimed",
      "text": "This scoped workflow result does not claim release readiness."
    },
    {
      "code": "route_assumptions_not_global_minimality",
      "text": "Route-required assumptions are not claimed to be globally minimal."
    }
  ],
  "evidence_ledger": {
    "version": "1.0",
    "scope": "scoped_high_level_workflow_result",
    "provenance": {
      "workflow": "assumptions_for",
      "status": "missing_assumptions",
      "certification_source": "none",
      "evidence_classes": [
        "missing_assumption"
      ]
    },
    "evidence_items": [
      {
        "id": "assumptions_for:missing-assumption",
        "class": "missing_assumption",
        "source": "route",
        "summary": "At least one route-required assumption is missing.",
        "low_level": {
          "status": "missing_assumptions",
          "reason": "At least one route-required assumption is missing.",
          "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
          "provided_assumptions": [],
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
              "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
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
            "status": "missing_assumptions",
            "reason": "At least one route-required assumption is missing.",
            "obligations": [
              {
                "id": "assumption-obligation-1",
                "lhs": "S S^T = A A^T for S = transpose(R), A^T = Q R",
                "rhs": "well-posed under route assumptions",
                "assumptions": [],
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
        }
      }
    ],
    "assumption_items": [
      {
        "text": "matrix dimensions are conformable for the operation",
        "status": "missing",
        "source": "matrix operation",
        "necessity": "required_by_route",
        "route_categories": [
          "shape_condition"
        ],
        "route_category_sources": [
          "assumption_rule:matrix_conformability"
        ],
        "used_by": [
          "matrix_conformability"
        ]
      }
    ],
    "veto_items": [],
    "action_items": [
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "non_claim_items": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      },
      {
        "code": "route_assumptions_not_global_minimality",
        "text": "Route-required assumptions are not claimed to be globally minimal."
      }
    ],
    "non_claim_codes": [
      "general_theorem_proving_not_claimed",
      "release_readiness_not_claimed",
      "route_assumptions_not_global_minimality"
    ],
    "boundary": "This ledger is case-local provenance for the same high-level workflow envelope. It is not independent proof, release evidence, public benchmark validation, or a claim of broad downstream-agent usefulness."
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "high_level_workflow_result"
  },
  "source": {
    "target": "S S^T = A A^T for S = transpose(R), A^T = Q R"
  },
  "coverage": {
    "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
    "provided_assumption_count": 0,
    "detected_assumption_count": 1,
    "missing_assumption_count": 1,
    "gap_count": 1,
    "proposal_count": 1,
    "inspected": [
      "direct_target"
    ],
    "not_inspected": [
      "global minimality",
      "full theorem applicability",
      "source-wide assumption consistency",
      "proof closure after adding assumptions"
    ]
  },
  "tool_uses": [
    {
      "tool": "assumptions_required",
      "arguments": {
        "target": "S S^T = A A^T for S = transpose(R), A^T = Q R",
        "provided_assumptions": []
      },
      "purpose": "Detect route-required assumptions with a bounded deterministic rule set.",
      "status": "completed",
      "output_contract": "assumption_discovery_result"
    },
    {
      "tool": "build_assumption_gaps",
      "arguments": {
        "target": "S S^T = A A^T for S = transpose(R), A^T = Q R"
      },
      "purpose": "Convert missing assumption records into localized gap objects.",
      "status": "completed",
      "output_contract": "assumption_gap_list"
    },
    {
      "tool": "build_assumption_proposals",
      "arguments": {
        "gap_count": "derived"
      },
      "purpose": "Create concrete assumption proposals linked to detected gaps.",
      "status": "completed",
      "output_contract": "assumption_proposal_list"
    }
  ],
  "gaps": [
    {
      "id": "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
      "location": "S S^T = A A^T for S = transpose(R), A^T = Q R",
      "problem": "Missing route-required assumption: matrix dimensions are conformable for the operation (matrix operation).",
      "why": "Matrix operations require shape, square, invertibility, or conformability conditions before the expression is well posed.",
      "affected_terms": [
        "S S^T = A A^T for S = transpose(R), A^T = Q R",
        "matrix_conformability"
      ],
      "route_categories": [
        "shape_condition"
      ],
      "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
      "source": "assumptions_required",
      "source_context": {},
      "evidence_refs": [
        "assumption_rule:matrix_conformability"
      ],
      "severity": "medium",
      "assumption": {
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
    }
  ],
  "proposals": [
    {
      "id": "assumption_proposal_assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
      "gap_ids": [
        "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation"
      ],
      "type": "add_assumption",
      "location": "S S^T = A A^T for S = transpose(R), A^T = Q R",
      "proposal_text": "Assume all matrix dimensions in this operation are conformable.",
      "rationale": "This proposal closes `assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation` by stating the route-required condition before using the affected expression.",
      "missing_assumptions": [
        "The target uses notation whose route rule requires this assumption before the expression is well posed."
      ],
      "possible_assumption_sets": [
        {
          "id": "route_rule_assumption",
          "role": "route condition",
          "assumptions": [
            "Assume all matrix dimensions in this operation are conformable."
          ],
          "closes": "States the deterministic route condition detected by the assumption rule."
        }
      ],
      "derivation_route": [
        {
          "step": "State route assumption",
          "detail": "Add or verify the detected assumption before applying the derivation step."
        }
      ],
      "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
      "evidence_refs": [
        "assumption_rule:matrix_conformability"
      ],
      "application_status": "not_applied",
      "validation": {
        "policy": "route_rule_non_certifying",
        "status": "validated_by_rule",
        "certifying": false,
        "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
        "backend_attempts": [
          {
            "backend": "assumption_rule",
            "status": "validated_by_rule",
            "severity": "diagnostic",
            "reason": "Matched route categories: shape_condition"
          }
        ],
        "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
        "gap_id": "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
        "proposal_id": "assumption_proposal_assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation"
      }
    }
  ],
  "validation": {
    "policy": "route_rule_non_certifying",
    "status_counts": {
      "validated_by_rule": 1
    },
    "proposal_count": 1,
    "backend_attempt_count": 1,
    "certifying": false,
    "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
  },
  "agent_handoff": {
    "scoped_question": "What assumptions are required for S S^T = A A^T for S = transpose(R), A^T = Q R?",
    "status": "missing_assumptions",
    "reason": "At least one route-required assumption is missing.",
    "source_context": "",
    "gap_count": 1,
    "proposal_count": 1,
    "assumption_gap_ledger": [
      {
        "id": "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
        "location": "S S^T = A A^T for S = transpose(R), A^T = Q R",
        "problem": "Missing route-required assumption: matrix dimensions are conformable for the operation (matrix operation).",
        "why": "Matrix operations require shape, square, invertibility, or conformability conditions before the expression is well posed.",
        "affected_terms": [
          "S S^T = A A^T for S = transpose(R), A^T = Q R",
          "matrix_conformability"
        ],
        "route_categories": [
          "shape_condition"
        ],
        "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
        "source": "assumptions_required",
        "source_context": {},
        "evidence_refs": [
          "assumption_rule:matrix_conformability"
        ],
        "severity": "medium",
        "assumption": {
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
      }
    ],
    "proposals": [
      {
        "id": "assumption_proposal_assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
        "gap_ids": [
          "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation"
        ],
        "type": "add_assumption",
        "location": "S S^T = A A^T for S = transpose(R), A^T = Q R",
        "proposal_text": "Assume all matrix dimensions in this operation are conformable.",
        "rationale": "This proposal closes `assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation` by stating the route-required condition before using the affected expression.",
        "missing_assumptions": [
          "The target uses notation whose route rule requires this assumption before the expression is well posed."
        ],
        "possible_assumption_sets": [
          {
            "id": "route_rule_assumption",
            "role": "route condition",
            "assumptions": [
              "Assume all matrix dimensions in this operation are conformable."
            ],
            "closes": "States the deterministic route condition detected by the assumption rule."
          }
        ],
        "derivation_route": [
          {
            "step": "State route assumption",
            "detail": "Add or verify the detected assumption before applying the derivation step."
          }
        ],
        "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
        "evidence_refs": [
          "assumption_rule:matrix_conformability"
        ],
        "application_status": "not_applied",
        "validation": {
          "policy": "route_rule_non_certifying",
          "status": "validated_by_rule",
          "certifying": false,
          "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
          "backend_attempts": [
            {
              "backend": "assumption_rule",
              "status": "validated_by_rule",
              "severity": "diagnostic",
              "reason": "Matched route categories: shape_condition"
            }
          ],
          "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
          "gap_id": "assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation",
          "proposal_id": "assumption_proposal_assumption_gap_direct_target_1_matrix_dimensions_are_conformable_for_the_operation"
        }
      }
    ],
    "validation": {
      "policy": "route_rule_non_certifying",
      "status_counts": {
        "validated_by_rule": 1
      },
      "proposal_count": 1,
      "backend_attempt_count": 1,
      "certifying": false,
      "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
    },
    "non_claim_boundary": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      },
      {
        "code": "route_assumptions_not_global_minimality",
        "text": "Route-required assumptions are not claimed to be globally minimal."
      }
    ],
    "next_actions": [
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "next_artifact": "Inspect the assumption proposals before editing text or retrying proof/derivation checks.",
    "certification_boundary": "Assumption proposals are diagnostic route repairs, not proof certificates, not applied edits, and not globally minimal assumption sets."
  }
}
=== Q2: sequential Cholesky downdate ===
{
  "status": "inconclusive",
  "workflow": "assumptions_for",
  "question": "What assumptions are required for L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate?",
  "claim_class": "assumption_discovery",
  "answer": "No route-required assumptions were detected by the bounded rule set.",
  "evidence": [
    {
      "id": "assumptions_for:human-review-required",
      "class": "human_review_required",
      "source": "kernel",
      "summary": "No route-required assumptions were detected by the bounded rule set.",
      "low_level": {
        "status": "unknown",
        "reason": "No route-required assumptions were detected by the bounded rule set.",
        "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
        "provided_assumptions": [],
        "assumptions": [],
        "missing_assumptions": [],
        "workbench_result": {
          "question": {
            "question_type": "assumptions_required",
            "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
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
              "lhs": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
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
      }
    }
  ],
  "evidence_classes": [
    "human_review_required"
  ],
  "certification_source": "none",
  "veto_reasons": [
    {
      "code": "unresolved_low_level_status",
      "reason": "Unsupported or inconclusive low-level status: unknown"
    }
  ],
  "assumptions": [],
  "counterexamples": [],
  "actions": [
    {
      "code": "human_review",
      "description": "Review low-level evidence manually."
    },
    {
      "code": "human_review",
      "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
    },
    {
      "code": "review_assumption_proposals",
      "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
    }
  ],
  "non_claims": [
    {
      "code": "general_theorem_proving_not_claimed",
      "text": "This scoped workflow result does not claim general theorem-proving ability."
    },
    {
      "code": "release_readiness_not_claimed",
      "text": "This scoped workflow result does not claim release readiness."
    }
  ],
  "evidence_ledger": {
    "version": "1.0",
    "scope": "scoped_high_level_workflow_result",
    "provenance": {
      "workflow": "assumptions_for",
      "status": "inconclusive",
      "certification_source": "none",
      "evidence_classes": [
        "human_review_required"
      ]
    },
    "evidence_items": [
      {
        "id": "assumptions_for:human-review-required",
        "class": "human_review_required",
        "source": "kernel",
        "summary": "No route-required assumptions were detected by the bounded rule set.",
        "low_level": {
          "status": "unknown",
          "reason": "No route-required assumptions were detected by the bounded rule set.",
          "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
          "provided_assumptions": [],
          "assumptions": [],
          "missing_assumptions": [],
          "workbench_result": {
            "question": {
              "question_type": "assumptions_required",
              "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
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
                "lhs": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
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
        }
      }
    ],
    "assumption_items": [],
    "veto_items": [
      {
        "code": "unresolved_low_level_status",
        "reason": "Unsupported or inconclusive low-level status: unknown"
      }
    ],
    "action_items": [
      {
        "code": "human_review",
        "description": "Review low-level evidence manually."
      },
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "non_claim_items": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      }
    ],
    "non_claim_codes": [
      "general_theorem_proving_not_claimed",
      "release_readiness_not_claimed"
    ],
    "boundary": "This ledger is case-local provenance for the same high-level workflow envelope. It is not independent proof, release evidence, public benchmark validation, or a claim of broad downstream-agent usefulness."
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "high_level_workflow_result"
  },
  "source": {
    "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate"
  },
  "coverage": {
    "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
    "provided_assumption_count": 0,
    "detected_assumption_count": 0,
    "missing_assumption_count": 0,
    "gap_count": 1,
    "proposal_count": 1,
    "inspected": [
      "direct_target"
    ],
    "not_inspected": [
      "global minimality",
      "full theorem applicability",
      "source-wide assumption consistency",
      "proof closure after adding assumptions"
    ]
  },
  "tool_uses": [
    {
      "tool": "assumptions_required",
      "arguments": {
        "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
        "provided_assumptions": []
      },
      "purpose": "Detect route-required assumptions with a bounded deterministic rule set.",
      "status": "completed",
      "output_contract": "assumption_discovery_result"
    },
    {
      "tool": "build_assumption_gaps",
      "arguments": {
        "target": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate"
      },
      "purpose": "Convert missing assumption records into localized gap objects.",
      "status": "completed",
      "output_contract": "assumption_gap_list"
    },
    {
      "tool": "build_assumption_proposals",
      "arguments": {
        "gap_count": "derived"
      },
      "purpose": "Create concrete assumption proposals linked to detected gaps.",
      "status": "completed",
      "output_contract": "assumption_proposal_list"
    }
  ],
  "gaps": [
    {
      "id": "assumption_gap_direct_target_unknown_route",
      "location": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
      "problem": "No route-required assumptions were detected by the bounded assumption rule set.",
      "why": "This is an evidence gap, not proof that no assumptions are needed. The target may require domain, shape, regularity, semantic, or source-backed assumptions outside the current rules.",
      "affected_terms": [
        "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate"
      ],
      "route_categories": [],
      "source": "assumptions_required",
      "evidence_refs": [
        "assumptions_required:bounded_rule_set_no_match"
      ],
      "severity": "low"
    }
  ],
  "proposals": [
    {
      "id": "assumption_proposal_assumption_gap_direct_target_unknown_route",
      "gap_ids": [
        "assumption_gap_direct_target_unknown_route"
      ],
      "type": "formalize_assumption",
      "location": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
      "proposal_text": "Formalize the target into a typed obligation or add a domain-specific assumption rule before claiming the assumption set is complete.",
      "rationale": "The bounded rule set could not identify route assumptions, so the next useful artifact is a typed target or domain-specific rule.",
      "missing_assumptions": [
        "The current bounded rules cannot identify a deterministic assumption route for this target.",
        "This is not evidence that no assumptions are needed; it means the target needs a typed obligation or domain-specific route rule."
      ],
      "possible_assumption_sets": [
        {
          "id": "typed_obligation_first",
          "role": "next deterministic artifact",
          "assumptions": [
            "Formalize the objects, domains, and operators in the target.",
            "Add domain-specific rules only after the formalized target identifies the relevant operations."
          ],
          "closes": "Makes the missing-assumption question inspectable by deterministic tools."
        }
      ],
      "derivation_route": [
        {
          "step": "Formalize target",
          "detail": "Convert the source expression into a typed obligation with explicit objects and operations."
        },
        {
          "step": "Run assumption discovery again",
          "detail": "Use the typed target or new domain rule to identify concrete route assumptions."
        }
      ],
      "evidence_refs": [
        "assumptions_required:bounded_rule_set_no_match"
      ],
      "application_status": "not_applied",
      "validation": {
        "policy": "route_rule_non_certifying",
        "status": "not_encodable",
        "certifying": false,
        "reason": "No deterministic assumption route was available for this target.",
        "backend_attempts": [
          {
            "backend": "assumption_rule",
            "status": "not_encodable",
            "severity": "diagnostic",
            "reason": "The bounded rule set did not match this target."
          }
        ],
        "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
        "gap_id": "assumption_gap_direct_target_unknown_route",
        "proposal_id": "assumption_proposal_assumption_gap_direct_target_unknown_route"
      }
    }
  ],
  "validation": {
    "policy": "route_rule_non_certifying",
    "status_counts": {
      "not_encodable": 1
    },
    "proposal_count": 1,
    "backend_attempt_count": 1,
    "certifying": false,
    "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
  },
  "agent_handoff": {
    "scoped_question": "What assumptions are required for L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate?",
    "status": "inconclusive",
    "reason": "No route-required assumptions were detected by the bounded rule set.",
    "source_context": "",
    "gap_count": 1,
    "proposal_count": 1,
    "assumption_gap_ledger": [
      {
        "id": "assumption_gap_direct_target_unknown_route",
        "location": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
        "problem": "No route-required assumptions were detected by the bounded assumption rule set.",
        "why": "This is an evidence gap, not proof that no assumptions are needed. The target may require domain, shape, regularity, semantic, or source-backed assumptions outside the current rules.",
        "affected_terms": [
          "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate"
        ],
        "route_categories": [],
        "source": "assumptions_required",
        "evidence_refs": [
          "assumptions_required:bounded_rule_set_no_match"
        ],
        "severity": "low"
      }
    ],
    "proposals": [
      {
        "id": "assumption_proposal_assumption_gap_direct_target_unknown_route",
        "gap_ids": [
          "assumption_gap_direct_target_unknown_route"
        ],
        "type": "formalize_assumption",
        "location": "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate",
        "proposal_text": "Formalize the target into a typed obligation or add a domain-specific assumption rule before claiming the assumption set is complete.",
        "rationale": "The bounded rule set could not identify route assumptions, so the next useful artifact is a typed target or domain-specific rule.",
        "missing_assumptions": [
          "The current bounded rules cannot identify a deterministic assumption route for this target.",
          "This is not evidence that no assumptions are needed; it means the target needs a typed obligation or domain-specific route rule."
        ],
        "possible_assumption_sets": [
          {
            "id": "typed_obligation_first",
            "role": "next deterministic artifact",
            "assumptions": [
              "Formalize the objects, domains, and operators in the target.",
              "Add domain-specific rules only after the formalized target identifies the relevant operations."
            ],
            "closes": "Makes the missing-assumption question inspectable by deterministic tools."
          }
        ],
        "derivation_route": [
          {
            "step": "Formalize target",
            "detail": "Convert the source expression into a typed obligation with explicit objects and operations."
          },
          {
            "step": "Run assumption discovery again",
            "detail": "Use the typed target or new domain rule to identify concrete route assumptions."
          }
        ],
        "evidence_refs": [
          "assumptions_required:bounded_rule_set_no_match"
        ],
        "application_status": "not_applied",
        "validation": {
          "policy": "route_rule_non_certifying",
          "status": "not_encodable",
          "certifying": false,
          "reason": "No deterministic assumption route was available for this target.",
          "backend_attempts": [
            {
              "backend": "assumption_rule",
              "status": "not_encodable",
              "severity": "diagnostic",
              "reason": "The bounded rule set did not match this target."
            }
          ],
          "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
          "gap_id": "assumption_gap_direct_target_unknown_route",
          "proposal_id": "assumption_proposal_assumption_gap_direct_target_unknown_route"
        }
      }
    ],
    "validation": {
      "policy": "route_rule_non_certifying",
      "status_counts": {
        "not_encodable": 1
      },
      "proposal_count": 1,
      "backend_attempt_count": 1,
      "certifying": false,
      "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
    },
    "non_claim_boundary": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      }
    ],
    "next_actions": [
      {
        "code": "human_review",
        "description": "Review low-level evidence manually."
      },
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "next_artifact": "Inspect the assumption proposals before editing text or retrying proof/derivation checks.",
    "certification_boundary": "Assumption proposals are diagnostic route repairs, not proof certificates, not applied edits, and not globally minimal assumption sets."
  }
}
=== Q3: logdet derivative ===
{
  "status": "missing_assumptions",
  "workflow": "assumptions_for",
  "question": "What assumptions are required for d logdet(S S^T) = 2 trace(S^{-1} dS)?",
  "claim_class": "assumption_discovery",
  "answer": "At least one route-required assumption is missing.",
  "evidence": [
    {
      "id": "assumptions_for:missing-assumption",
      "class": "missing_assumption",
      "source": "route",
      "summary": "At least one route-required assumption is missing.",
      "low_level": {
        "status": "missing_assumptions",
        "reason": "At least one route-required assumption is missing.",
        "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "provided_assumptions": [],
        "assumptions": [
          {
            "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
            "status": "missing",
            "source": "determinant or logdet",
            "necessity": "required_by_route",
            "used_by": [
              "logdet_domain"
            ],
            "route_categories": [
              "covariance_condition",
              "domain_condition"
            ],
            "route_category_sources": [
              "assumption_rule:logdet_domain"
            ]
          },
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
            "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
            "status": "missing",
            "source": "determinant or logdet",
            "necessity": "required_by_route",
            "used_by": [
              "logdet_domain"
            ],
            "route_categories": [
              "covariance_condition",
              "domain_condition"
            ],
            "route_category_sources": [
              "assumption_rule:logdet_domain"
            ]
          },
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
            "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
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
          "status": "missing_assumptions",
          "reason": "At least one route-required assumption is missing.",
          "obligations": [
            {
              "id": "assumption-obligation-1",
              "lhs": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
              "rhs": "well-posed under route assumptions",
              "assumptions": [],
              "status": "missing_assumptions",
              "reason": "At least one route-required assumption is missing.",
              "backend_attempts": [],
              "counterexample": null,
              "missing_assumptions": [
                {
                  "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
                  "status": "missing",
                  "source": "determinant or logdet",
                  "necessity": "required_by_route",
                  "used_by": [
                    "logdet_domain"
                  ],
                  "route_categories": [
                    "covariance_condition",
                    "domain_condition"
                  ],
                  "route_category_sources": [
                    "assumption_rule:logdet_domain"
                  ]
                },
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
              "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
              "status": "missing",
              "source": "determinant or logdet",
              "necessity": "required_by_route",
              "used_by": [
                "logdet_domain"
              ],
              "route_categories": [
                "covariance_condition",
                "domain_condition"
              ],
              "route_category_sources": [
                "assumption_rule:logdet_domain"
              ]
            },
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
              "assumption": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
              "source": "determinant or logdet"
            },
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
      }
    }
  ],
  "evidence_classes": [
    "missing_assumption"
  ],
  "certification_source": "none",
  "veto_reasons": [],
  "assumptions": [
    {
      "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
      "status": "missing",
      "source": "determinant or logdet",
      "necessity": "required_by_route",
      "used_by": [
        "logdet_domain"
      ],
      "route_categories": [
        "covariance_condition",
        "domain_condition"
      ],
      "route_category_sources": [
        "assumption_rule:logdet_domain"
      ]
    },
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
  "counterexamples": [],
  "actions": [
    {
      "code": "human_review",
      "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
    },
    {
      "code": "review_assumption_proposals",
      "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
    }
  ],
  "non_claims": [
    {
      "code": "general_theorem_proving_not_claimed",
      "text": "This scoped workflow result does not claim general theorem-proving ability."
    },
    {
      "code": "release_readiness_not_claimed",
      "text": "This scoped workflow result does not claim release readiness."
    },
    {
      "code": "route_assumptions_not_global_minimality",
      "text": "Route-required assumptions are not claimed to be globally minimal."
    }
  ],
  "evidence_ledger": {
    "version": "1.0",
    "scope": "scoped_high_level_workflow_result",
    "provenance": {
      "workflow": "assumptions_for",
      "status": "missing_assumptions",
      "certification_source": "none",
      "evidence_classes": [
        "missing_assumption"
      ]
    },
    "evidence_items": [
      {
        "id": "assumptions_for:missing-assumption",
        "class": "missing_assumption",
        "source": "route",
        "summary": "At least one route-required assumption is missing.",
        "low_level": {
          "status": "missing_assumptions",
          "reason": "At least one route-required assumption is missing.",
          "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
          "provided_assumptions": [],
          "assumptions": [
            {
              "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
              "status": "missing",
              "source": "determinant or logdet",
              "necessity": "required_by_route",
              "used_by": [
                "logdet_domain"
              ],
              "route_categories": [
                "covariance_condition",
                "domain_condition"
              ],
              "route_category_sources": [
                "assumption_rule:logdet_domain"
              ]
            },
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
              "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
              "status": "missing",
              "source": "determinant or logdet",
              "necessity": "required_by_route",
              "used_by": [
                "logdet_domain"
              ],
              "route_categories": [
                "covariance_condition",
                "domain_condition"
              ],
              "route_category_sources": [
                "assumption_rule:logdet_domain"
              ]
            },
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
              "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
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
            "status": "missing_assumptions",
            "reason": "At least one route-required assumption is missing.",
            "obligations": [
              {
                "id": "assumption-obligation-1",
                "lhs": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
                "rhs": "well-posed under route assumptions",
                "assumptions": [],
                "status": "missing_assumptions",
                "reason": "At least one route-required assumption is missing.",
                "backend_attempts": [],
                "counterexample": null,
                "missing_assumptions": [
                  {
                    "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
                    "status": "missing",
                    "source": "determinant or logdet",
                    "necessity": "required_by_route",
                    "used_by": [
                      "logdet_domain"
                    ],
                    "route_categories": [
                      "covariance_condition",
                      "domain_condition"
                    ],
                    "route_category_sources": [
                      "assumption_rule:logdet_domain"
                    ]
                  },
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
                "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
                "status": "missing",
                "source": "determinant or logdet",
                "necessity": "required_by_route",
                "used_by": [
                  "logdet_domain"
                ],
                "route_categories": [
                  "covariance_condition",
                  "domain_condition"
                ],
                "route_category_sources": [
                  "assumption_rule:logdet_domain"
                ]
              },
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
                "assumption": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
                "source": "determinant or logdet"
              },
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
        }
      }
    ],
    "assumption_items": [
      {
        "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
        "status": "missing",
        "source": "determinant or logdet",
        "necessity": "required_by_route",
        "route_categories": [
          "covariance_condition",
          "domain_condition"
        ],
        "route_category_sources": [
          "assumption_rule:logdet_domain"
        ],
        "used_by": [
          "logdet_domain"
        ]
      },
      {
        "text": "matrix dimensions are conformable for the operation",
        "status": "missing",
        "source": "matrix operation",
        "necessity": "required_by_route",
        "route_categories": [
          "shape_condition"
        ],
        "route_category_sources": [
          "assumption_rule:matrix_conformability"
        ],
        "used_by": [
          "matrix_conformability"
        ]
      }
    ],
    "veto_items": [],
    "action_items": [
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "non_claim_items": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      },
      {
        "code": "route_assumptions_not_global_minimality",
        "text": "Route-required assumptions are not claimed to be globally minimal."
      }
    ],
    "non_claim_codes": [
      "general_theorem_proving_not_claimed",
      "release_readiness_not_claimed",
      "route_assumptions_not_global_minimality"
    ],
    "boundary": "This ledger is case-local provenance for the same high-level workflow envelope. It is not independent proof, release evidence, public benchmark validation, or a claim of broad downstream-agent usefulness."
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "high_level_workflow_result"
  },
  "source": {
    "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)"
  },
  "coverage": {
    "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
    "provided_assumption_count": 0,
    "detected_assumption_count": 2,
    "missing_assumption_count": 2,
    "gap_count": 2,
    "proposal_count": 2,
    "inspected": [
      "direct_target"
    ],
    "not_inspected": [
      "global minimality",
      "full theorem applicability",
      "source-wide assumption consistency",
      "proof closure after adding assumptions"
    ]
  },
  "tool_uses": [
    {
      "tool": "assumptions_required",
      "arguments": {
        "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "provided_assumptions": []
      },
      "purpose": "Detect route-required assumptions with a bounded deterministic rule set.",
      "status": "completed",
      "output_contract": "assumption_discovery_result"
    },
    {
      "tool": "build_assumption_gaps",
      "arguments": {
        "target": "d logdet(S S^T) = 2 trace(S^{-1} dS)"
      },
      "purpose": "Convert missing assumption records into localized gap objects.",
      "status": "completed",
      "output_contract": "assumption_gap_list"
    },
    {
      "tool": "build_assumption_proposals",
      "arguments": {
        "gap_count": "derived"
      },
      "purpose": "Create concrete assumption proposals linked to detected gaps.",
      "status": "completed",
      "output_contract": "assumption_proposal_list"
    }
  ],
  "gaps": [
    {
      "id": "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
      "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
      "problem": "Missing route-required assumption: matrix operand is square with valid determinant domain, usually positive definite for logdet (determinant or logdet).",
      "why": "Determinant and logdet notation require a valid determinant domain; covariance-style uses usually require positive definiteness.",
      "affected_terms": [
        "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "logdet_domain"
      ],
      "route_categories": [
        "covariance_condition",
        "domain_condition"
      ],
      "route_kind": "matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
      "source": "assumptions_required",
      "source_context": {},
      "evidence_refs": [
        "assumption_rule:logdet_domain"
      ],
      "severity": "medium",
      "assumption": {
        "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
        "status": "missing",
        "source": "determinant or logdet",
        "necessity": "required_by_route",
        "used_by": [
          "logdet_domain"
        ],
        "route_categories": [
          "covariance_condition",
          "domain_condition"
        ],
        "route_category_sources": [
          "assumption_rule:logdet_domain"
        ]
      }
    },
    {
      "id": "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
      "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
      "problem": "Missing route-required assumption: matrix dimensions are conformable for the operation (matrix operation).",
      "why": "Matrix operations require shape, square, invertibility, or conformability conditions before the expression is well posed.",
      "affected_terms": [
        "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "matrix_conformability"
      ],
      "route_categories": [
        "shape_condition"
      ],
      "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
      "source": "assumptions_required",
      "source_context": {},
      "evidence_refs": [
        "assumption_rule:matrix_conformability"
      ],
      "severity": "medium",
      "assumption": {
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
    }
  ],
  "proposals": [
    {
      "id": "assumption_proposal_assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
      "gap_ids": [
        "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet"
      ],
      "type": "add_assumption",
      "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
      "proposal_text": "Assume the matrix operand is square and has a valid determinant domain; for covariance/logdet use, assume it is positive definite.",
      "rationale": "This proposal closes `assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet` by stating the route-required condition before using the affected expression.",
      "missing_assumptions": [
        "The target uses notation whose route rule requires this assumption before the expression is well posed."
      ],
      "possible_assumption_sets": [
        {
          "id": "route_rule_assumption",
          "role": "route condition",
          "assumptions": [
            "Assume the matrix operand is square and has a valid determinant domain; for covariance/logdet use, assume it is positive definite."
          ],
          "closes": "States the deterministic route condition detected by the assumption rule."
        }
      ],
      "derivation_route": [
        {
          "step": "State route assumption",
          "detail": "Add or verify the detected assumption before applying the derivation step."
        }
      ],
      "route_kind": "matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
      "evidence_refs": [
        "assumption_rule:logdet_domain"
      ],
      "application_status": "not_applied",
      "validation": {
        "policy": "route_rule_non_certifying",
        "status": "validated_by_rule",
        "certifying": false,
        "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
        "backend_attempts": [
          {
            "backend": "assumption_rule",
            "status": "validated_by_rule",
            "severity": "diagnostic",
            "reason": "Matched route categories: covariance_condition, domain_condition"
          }
        ],
        "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
        "gap_id": "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
        "proposal_id": "assumption_proposal_assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet"
      }
    },
    {
      "id": "assumption_proposal_assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
      "gap_ids": [
        "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation"
      ],
      "type": "add_assumption",
      "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
      "proposal_text": "Assume all matrix dimensions in this operation are conformable.",
      "rationale": "This proposal closes `assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation` by stating the route-required condition before using the affected expression.",
      "missing_assumptions": [
        "The target uses notation whose route rule requires this assumption before the expression is well posed."
      ],
      "possible_assumption_sets": [
        {
          "id": "route_rule_assumption",
          "role": "route condition",
          "assumptions": [
            "Assume all matrix dimensions in this operation are conformable."
          ],
          "closes": "States the deterministic route condition detected by the assumption rule."
        }
      ],
      "derivation_route": [
        {
          "step": "State route assumption",
          "detail": "Add or verify the detected assumption before applying the derivation step."
        }
      ],
      "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
      "evidence_refs": [
        "assumption_rule:matrix_conformability"
      ],
      "application_status": "not_applied",
      "validation": {
        "policy": "route_rule_non_certifying",
        "status": "validated_by_rule",
        "certifying": false,
        "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
        "backend_attempts": [
          {
            "backend": "assumption_rule",
            "status": "validated_by_rule",
            "severity": "diagnostic",
            "reason": "Matched route categories: shape_condition"
          }
        ],
        "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
        "gap_id": "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
        "proposal_id": "assumption_proposal_assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation"
      }
    }
  ],
  "validation": {
    "policy": "route_rule_non_certifying",
    "status_counts": {
      "validated_by_rule": 2
    },
    "proposal_count": 2,
    "backend_attempt_count": 2,
    "certifying": false,
    "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
  },
  "agent_handoff": {
    "scoped_question": "What assumptions are required for d logdet(S S^T) = 2 trace(S^{-1} dS)?",
    "status": "missing_assumptions",
    "reason": "At least one route-required assumption is missing.",
    "source_context": "",
    "gap_count": 2,
    "proposal_count": 2,
    "assumption_gap_ledger": [
      {
        "id": "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
        "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "problem": "Missing route-required assumption: matrix operand is square with valid determinant domain, usually positive definite for logdet (determinant or logdet).",
        "why": "Determinant and logdet notation require a valid determinant domain; covariance-style uses usually require positive definiteness.",
        "affected_terms": [
          "d logdet(S S^T) = 2 trace(S^{-1} dS)",
          "logdet_domain"
        ],
        "route_categories": [
          "covariance_condition",
          "domain_condition"
        ],
        "route_kind": "matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
        "source": "assumptions_required",
        "source_context": {},
        "evidence_refs": [
          "assumption_rule:logdet_domain"
        ],
        "severity": "medium",
        "assumption": {
          "text": "matrix operand is square with valid determinant domain, usually positive definite for logdet",
          "status": "missing",
          "source": "determinant or logdet",
          "necessity": "required_by_route",
          "used_by": [
            "logdet_domain"
          ],
          "route_categories": [
            "covariance_condition",
            "domain_condition"
          ],
          "route_category_sources": [
            "assumption_rule:logdet_domain"
          ]
        }
      },
      {
        "id": "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
        "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "problem": "Missing route-required assumption: matrix dimensions are conformable for the operation (matrix operation).",
        "why": "Matrix operations require shape, square, invertibility, or conformability conditions before the expression is well posed.",
        "affected_terms": [
          "d logdet(S S^T) = 2 trace(S^{-1} dS)",
          "matrix_conformability"
        ],
        "route_categories": [
          "shape_condition"
        ],
        "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
        "source": "assumptions_required",
        "source_context": {},
        "evidence_refs": [
          "assumption_rule:matrix_conformability"
        ],
        "severity": "medium",
        "assumption": {
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
      }
    ],
    "proposals": [
      {
        "id": "assumption_proposal_assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
        "gap_ids": [
          "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet"
        ],
        "type": "add_assumption",
        "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "proposal_text": "Assume the matrix operand is square and has a valid determinant domain; for covariance/logdet use, assume it is positive definite.",
        "rationale": "This proposal closes `assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet` by stating the route-required condition before using the affected expression.",
        "missing_assumptions": [
          "The target uses notation whose route rule requires this assumption before the expression is well posed."
        ],
        "possible_assumption_sets": [
          {
            "id": "route_rule_assumption",
            "role": "route condition",
            "assumptions": [
              "Assume the matrix operand is square and has a valid determinant domain; for covariance/logdet use, assume it is positive definite."
            ],
            "closes": "States the deterministic route condition detected by the assumption rule."
          }
        ],
        "derivation_route": [
          {
            "step": "State route assumption",
            "detail": "Add or verify the detected assumption before applying the derivation step."
          }
        ],
        "route_kind": "matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
        "evidence_refs": [
          "assumption_rule:logdet_domain"
        ],
        "application_status": "not_applied",
        "validation": {
          "policy": "route_rule_non_certifying",
          "status": "validated_by_rule",
          "certifying": false,
          "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
          "backend_attempts": [
            {
              "backend": "assumption_rule",
              "status": "validated_by_rule",
              "severity": "diagnostic",
              "reason": "Matched route categories: covariance_condition, domain_condition"
            }
          ],
          "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
          "gap_id": "assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet",
          "proposal_id": "assumption_proposal_assumption_gap_direct_target_1_matrix_operand_is_square_with_valid_determinant_domain_usually_positive_definite_for_logdet"
        }
      },
      {
        "id": "assumption_proposal_assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
        "gap_ids": [
          "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation"
        ],
        "type": "add_assumption",
        "location": "d logdet(S S^T) = 2 trace(S^{-1} dS)",
        "proposal_text": "Assume all matrix dimensions in this operation are conformable.",
        "rationale": "This proposal closes `assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation` by stating the route-required condition before using the affected expression.",
        "missing_assumptions": [
          "The target uses notation whose route rule requires this assumption before the expression is well posed."
        ],
        "possible_assumption_sets": [
          {
            "id": "route_rule_assumption",
            "role": "route condition",
            "assumptions": [
              "Assume all matrix dimensions in this operation are conformable."
            ],
            "closes": "States the deterministic route condition detected by the assumption rule."
          }
        ],
        "derivation_route": [
          {
            "step": "State route assumption",
            "detail": "Add or verify the detected assumption before applying the derivation step."
          }
        ],
        "route_kind": "matrix_dimensions_are_conformable_for_the_operation",
        "evidence_refs": [
          "assumption_rule:matrix_conformability"
        ],
        "application_status": "not_applied",
        "validation": {
          "policy": "route_rule_non_certifying",
          "status": "validated_by_rule",
          "certifying": false,
          "reason": "The proposal was derived from the deterministic assumption route rule for this gap.",
          "backend_attempts": [
            {
              "backend": "assumption_rule",
              "status": "validated_by_rule",
              "severity": "diagnostic",
              "reason": "Matched route categories: shape_condition"
            }
          ],
          "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality.",
          "gap_id": "assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation",
          "proposal_id": "assumption_proposal_assumption_gap_direct_target_2_matrix_dimensions_are_conformable_for_the_operation"
        }
      }
    ],
    "validation": {
      "policy": "route_rule_non_certifying",
      "status_counts": {
        "validated_by_rule": 2
      },
      "proposal_count": 2,
      "backend_attempt_count": 2,
      "certifying": false,
      "boundary": "Rule validation only checks that the proposed assumption matches a deterministic route requirement; it is not a proof certificate and does not prove global minimality."
    },
    "non_claim_boundary": [
      {
        "code": "general_theorem_proving_not_claimed",
        "text": "This scoped workflow result does not claim general theorem-proving ability."
      },
      {
        "code": "release_readiness_not_claimed",
        "text": "This scoped workflow result does not claim release readiness."
      },
      {
        "code": "route_assumptions_not_global_minimality",
        "text": "Route-required assumptions are not claimed to be globally minimal."
      }
    ],
    "next_actions": [
      {
        "code": "human_review",
        "description": "Review whether route-required assumptions are sufficient for the intended mathematical setting."
      },
      {
        "code": "review_assumption_proposals",
        "description": "Inspect gap-linked assumption proposals before editing the document or retrying a derivation."
      }
    ],
    "next_artifact": "Inspect the assumption proposals before editing text or retrying proof/derivation checks.",
    "certification_boundary": "Assumption proposals are diagnostic route repairs, not proof certificates, not applied edits, and not globally minimal assumption sets."
  }
}
