=== C1: audit-kalman-recursion kalman_qr_tf.py ===
{
  "status": "mismatch",
  "reason": "Kalman recursion implementation is missing required structural operations.",
  "required_operations": [
    "qr",
    "triangular_solve"
  ],
  "observed_operations": [
    "assignment",
    "call",
    "cholesky",
    "covariance_update",
    "expectation",
    "innovation_covariance",
    "innovation_update",
    "kalman_gain",
    "loop",
    "matmul",
    "posterior_or_likelihood",
    "prediction_update",
    "quadratic_form",
    "reparameterization_gradient",
    "return",
    "shape_guard",
    "shape_reference",
    "state_update",
    "subscript",
    "time_step_update"
  ],
  "missing_operations": [
    "qr",
    "triangular_solve"
  ],
  "shape_diagnostics": {
    "status": "missing_guards",
    "reason": "Some expected shape/covariance guards were not found in the Python AST.",
    "required_guards": [
      "shape_guard",
      "covariance_guard"
    ],
    "missing_guards": [
      "covariance_guard"
    ],
    "evidence": [
      {
        "operation": "shape_guard",
        "line": 73,
        "expression": "y.shape.rank == 1"
      },
      {
        "operation": "shape_guard",
        "line": 75,
        "expression": "y.shape.rank != 2"
      },
      {
        "operation": "shape_guard",
        "line": 81,
        "expression": "matrix.shape.rank == 3"
      },
      {
        "operation": "shape_guard",
        "line": 87,
        "expression": "vector.shape.rank == 2"
      },
      {
        "operation": "shape_guard",
        "line": 112,
        "expression": "y.shape.rank != 2"
      },
      {
        "operation": "shape_guard",
        "line": 141,
        "expression": "tensor.shape.rank != rank"
      },
      {
        "operation": "shape_guard",
        "line": 178,
        "expression": "covariance.shape.rank != 3"
      },
      {
        "operation": "shape_guard",
        "line": 522,
        "expression": "transition_noise_factor.shape.rank != 2"
      },
      {
        "operation": "shape_guard",
        "line": 524,
        "expression": "initial_state_factor.shape.rank != 2"
      }
    ],
    "metadata": {
      "schema_version": "1.0",
      "contract": "kalman_shape_diagnostics"
    }
  },
  "ast_operation_graph": {
    "status": "consistent",
    "reason": "AST operation graph extracted from Python source.",
    "source_path": "bayesfilter/linear/kalman_qr_tf.py",
    "operations": [
      "assignment",
      "call",
      "cholesky",
      "covariance_update",
      "expectation",
      "innovation_covariance",
      "innovation_update",
      "kalman_gain",
      "loop",
      "matmul",
      "posterior_or_likelihood",
      "prediction_update",
      "quadratic_form",
      "reparameterization_gradient",
      "return",
      "shape_guard",
      "shape_reference",
      "state_update",
      "subscript",
      "time_step_update"
    ],
    "nodes": [
      {
        "id": "22:0:assign:assignment:0",
        "kind": "assign",
        "operation": "assignment",
        "target": "TFQRLinearValueBackend",
        "expression": "Literal['tf_qr', 'tf_masked_qr']",
        "line": 22,
        "column": 0,
        "evidence": {
          "targets": [
            "TFQRLinearValueBackend"
          ]
        }
      },
      {
        "id": "22:0:assign:kalman_gain:1",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "TFQRLinearValueBackend",
        "expression": "Literal['tf_qr', 'tf_masked_qr']",
        "line": 22,
        "column": 0,
        "evidence": {
          "targets": [
            "TFQRLinearValueBackend"
          ]
        }
      },
      {
        "id": "22:25:subscript:subscript:2",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "Literal['tf_qr', 'tf_masked_qr']",
        "line": 22,
        "column": 25,
        "evidence": {}
      },
      {
        "id": "26:4:return:return:3",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "as_float_tensor(value, dtype)",
        "line": 26,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "26:11:call:call:4",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "as_float_tensor(value, dtype)",
        "line": 26,
        "column": 11,
        "evidence": {
          "function": "as_float_tensor"
        }
      },
      {
        "id": "43:4:return:return:5",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context=context)",
        "line": 43,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "43:11:call:call:6",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context=context)",
        "line": 43,
        "column": 11,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "59:4:return:return:7",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf.linalg.matvec(matrix, vector)",
        "line": 59,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "59:11:call:call:8",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.matvec(matrix, vector)",
        "line": 59,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.matvec"
        }
      },
      {
        "id": "63:4:return:return:9",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf.linalg.matrix_transpose(matrix)",
        "line": 63,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "63:11:call:call:10",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.matrix_transpose(matrix)",
        "line": 63,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.matrix_transpose"
        }
      },
      {
        "id": "71:8:assign:assignment:11",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "common_floating_dtype(observations, context='observations')",
        "line": 71,
        "column": 8,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "71:8:assign:time_step_update:12",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "common_floating_dtype(observations, context='observations')",
        "line": 71,
        "column": 8,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "71:16:call:call:13",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(observations, context='observations')",
        "line": 71,
        "column": 16,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "72:4:assign:assignment:14",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "as_float_tensor(observations, dtype, name='observations')",
        "line": 72,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "72:4:assign:time_step_update:15",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "as_float_tensor(observations, dtype, name='observations')",
        "line": 72,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "72:8:call:call:16",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "as_float_tensor(observations, dtype, name='observations')",
        "line": 72,
        "column": 8,
        "evidence": {
          "function": "as_float_tensor"
        }
      },
      {
        "id": "73:7:compare:shape_guard:17",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "y.shape.rank == 1",
        "line": 73,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "74:8:assign:assignment:18",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "y[:, tf.newaxis]",
        "line": 74,
        "column": 8,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "74:12:subscript:subscript:19",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[:, tf.newaxis]",
        "line": 74,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "75:7:compare:shape_guard:20",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "y.shape.rank != 2",
        "line": 75,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "76:14:call:call:21",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('observations must be one- or two-dimensional')",
        "line": 76,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "77:4:return:return:22",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "y",
        "line": 77,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "81:7:compare:shape_guard:23",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "matrix.shape.rank == 3",
        "line": 81,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "82:8:return:return:24",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "matrix[time_index]",
        "line": 82,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "82:15:subscript:subscript:25",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "matrix[time_index]",
        "line": 82,
        "column": 15,
        "evidence": {}
      },
      {
        "id": "83:4:return:return:26",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "matrix",
        "line": 83,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "87:7:compare:shape_guard:27",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "vector.shape.rank == 2",
        "line": 87,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "88:8:return:return:28",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "vector[time_index]",
        "line": 88,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "88:15:subscript:subscript:29",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "vector[time_index]",
        "line": 88,
        "column": 15,
        "evidence": {}
      },
      {
        "id": "89:4:return:return:30",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "vector",
        "line": 89,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "93:4:call:call:31",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.debugging.assert_equal(tf.shape(observation_mask), tf.shape(observations), message='Observation mask shape must match observations shape.')",
        "line": 93,
        "column": 4,
        "evidence": {
          "function": "tf.debugging.assert_equal"
        }
      },
      {
        "id": "94:8:call:call:32",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observation_mask)",
        "line": 94,
        "column": 8,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "95:8:call:call:33",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observations)",
        "line": 95,
        "column": 8,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "101:4:assign:assignment:34",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "observations.shape[0]",
        "line": 101,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "101:4:assign:innovation_covariance:35",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "n_timesteps",
        "expression": "observations.shape[0]",
        "line": 101,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "101:4:assign:reparameterization_gradient:36",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "observations.shape[0]",
        "line": 101,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "101:4:assign:shape_reference:37",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "n_timesteps",
        "expression": "observations.shape[0]",
        "line": 101,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "101:18:subscript:subscript:38",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "observations.shape[0]",
        "line": 101,
        "column": 18,
        "evidence": {}
      },
      {
        "id": "103:14:call:call:39",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('QR square-root filters require a static observation length')",
        "line": 103,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "104:4:return:return:40",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "int(n_timesteps)",
        "line": 104,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "104:11:call:call:41",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "int(n_timesteps)",
        "line": 104,
        "column": 11,
        "evidence": {
          "function": "int"
        }
      },
      {
        "id": "111:4:assign:assignment:42",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 111,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "111:4:assign:time_step_update:43",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 111,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "111:8:call:call:44",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 111,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "112:7:compare:shape_guard:45",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "y.shape.rank != 2",
        "line": 112,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "113:14:call:call:46",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('batched-static observations must have shape [time, observation]')",
        "line": 113,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "114:4:return:return:47",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "y",
        "line": 114,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "130:4:assign:assignment:48",
        "kind": "assign",
        "operation": "assignment",
        "target": "expected_ranks",
        "expression": "{'initial_state_mean': (initial_state_mean, 2), 'initial_state_covariance': (initial_state_covariance, 3), 'transition_offset': (transition_offset, 2), 'transition_matrix': (transition_matrix, 3), 'transition_covariance': (transition_covariance, 3), 'observation_offset': (observation_offset, 2), 'observation_matrix': (observation_matrix, 3), 'observation_covariance': (observation_covariance, 3)}",
        "line": 130,
        "column": 4,
        "evidence": {
          "targets": [
            "expected_ranks"
          ]
        }
      },
      {
        "id": "130:4:assign:expectation:49",
        "kind": "assign",
        "operation": "expectation",
        "target": "expected_ranks",
        "expression": "{'initial_state_mean': (initial_state_mean, 2), 'initial_state_covariance': (initial_state_covariance, 3), 'transition_offset': (transition_offset, 2), 'transition_matrix': (transition_matrix, 3), 'transition_covariance': (transition_covariance, 3), 'observation_offset': (observation_offset, 2), 'observation_matrix': (observation_matrix, 3), 'observation_covariance': (observation_covariance, 3)}",
        "line": 130,
        "column": 4,
        "evidence": {
          "targets": [
            "expected_ranks"
          ]
        }
      },
      {
        "id": "130:4:assign:innovation_covariance:50",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "expected_ranks",
        "expression": "{'initial_state_mean': (initial_state_mean, 2), 'initial_state_covariance': (initial_state_covariance, 3), 'transition_offset': (transition_offset, 2), 'transition_matrix': (transition_matrix, 3), 'transition_covariance': (transition_covariance, 3), 'observation_offset': (observation_offset, 2), 'observation_matrix': (observation_matrix, 3), 'observation_covariance': (observation_covariance, 3)}",
        "line": 130,
        "column": 4,
        "evidence": {
          "targets": [
            "expected_ranks"
          ]
        }
      },
      {
        "id": "130:4:assign:kalman_gain:51",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "expected_ranks",
        "expression": "{'initial_state_mean': (initial_state_mean, 2), 'initial_state_covariance': (initial_state_covariance, 3), 'transition_offset': (transition_offset, 2), 'transition_matrix': (transition_matrix, 3), 'transition_covariance': (transition_covariance, 3), 'observation_offset': (observation_offset, 2), 'observation_matrix': (observation_matrix, 3), 'observation_covariance': (observation_covariance, 3)}",
        "line": 130,
        "column": 4,
        "evidence": {
          "targets": [
            "expected_ranks"
          ]
        }
      },
      {
        "id": "140:4:loop:loop:52",
        "kind": "loop",
        "operation": "loop",
        "target": "(name, (tensor, rank))",
        "expression": "expected_ranks.items()",
        "line": 140,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "140:32:call:call:53",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "expected_ranks.items()",
        "line": 140,
        "column": 32,
        "evidence": {
          "function": "expected_ranks.items"
        }
      },
      {
        "id": "141:11:compare:shape_guard:54",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "tensor.shape.rank != rank",
        "line": 141,
        "column": 11,
        "evidence": {}
      },
      {
        "id": "142:18:call:call:55",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError(f'{name} must have rank {rank} for batched-static QR')",
        "line": 142,
        "column": 18,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "148:4:assign:assignment:56",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "common_floating_dtype(matrix, context='batched QR matrix')",
        "line": 148,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "148:4:assign:time_step_update:57",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "common_floating_dtype(matrix, context='batched QR matrix')",
        "line": 148,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "148:12:call:call:58",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(matrix, context='batched QR matrix')",
        "line": 148,
        "column": 12,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "149:4:assign:assignment:59",
        "kind": "assign",
        "operation": "assignment",
        "target": "matrix",
        "expression": "as_float_tensor(matrix, dtype, name='matrix')",
        "line": 149,
        "column": 4,
        "evidence": {
          "targets": [
            "matrix"
          ]
        }
      },
      {
        "id": "149:4:assign:time_step_update:60",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "matrix",
        "expression": "as_float_tensor(matrix, dtype, name='matrix')",
        "line": 149,
        "column": 4,
        "evidence": {
          "targets": [
            "matrix"
          ]
        }
      },
      {
        "id": "149:13:call:call:61",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "as_float_tensor(matrix, dtype, name='matrix')",
        "line": 149,
        "column": 13,
        "evidence": {
          "function": "as_float_tensor"
        }
      },
      {
        "id": "150:4:assign:assignment:62",
        "kind": "assign",
        "operation": "assignment",
        "target": "q, r",
        "expression": "tf.linalg.qr(matrix, full_matrices=False)",
        "line": 150,
        "column": 4,
        "evidence": {
          "targets": [
            "q",
            "r"
          ]
        }
      },
      {
        "id": "150:11:call:call:63",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.qr(matrix, full_matrices=False)",
        "line": 150,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.qr"
        }
      },
      {
        "id": "151:4:assign:assignment:64",
        "kind": "assign",
        "operation": "assignment",
        "target": "signs",
        "expression": "tf.sign(tf.linalg.diag_part(r))",
        "line": 151,
        "column": 4,
        "evidence": {
          "targets": [
            "signs"
          ]
        }
      },
      {
        "id": "151:4:assign:innovation_covariance:65",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "signs",
        "expression": "tf.sign(tf.linalg.diag_part(r))",
        "line": 151,
        "column": 4,
        "evidence": {
          "targets": [
            "signs"
          ]
        }
      },
      {
        "id": "151:12:call:call:66",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.sign(tf.linalg.diag_part(r))",
        "line": 151,
        "column": 12,
        "evidence": {
          "function": "tf.sign"
        }
      },
      {
        "id": "151:20:call:call:67",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(r)",
        "line": 151,
        "column": 20,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "152:4:assign:assignment:68",
        "kind": "assign",
        "operation": "assignment",
        "target": "signs",
        "expression": "tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)",
        "line": 152,
        "column": 4,
        "evidence": {
          "targets": [
            "signs"
          ]
        }
      },
      {
        "id": "152:4:assign:innovation_covariance:69",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "signs",
        "expression": "tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)",
        "line": 152,
        "column": 4,
        "evidence": {
          "targets": [
            "signs"
          ]
        }
      },
      {
        "id": "152:12:call:call:70",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)",
        "line": 152,
        "column": 12,
        "evidence": {
          "function": "tf.where"
        }
      },
      {
        "id": "152:21:call:call:71",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.equal(signs, 0.0)",
        "line": 152,
        "column": 21,
        "evidence": {
          "function": "tf.equal"
        }
      },
      {
        "id": "152:43:call:call:72",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.ones_like(signs)",
        "line": 152,
        "column": 43,
        "evidence": {
          "function": "tf.ones_like"
        }
      },
      {
        "id": "153:4:return:return:73",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(q * signs[..., tf.newaxis, :], signs[..., :, tf.newaxis] * r)",
        "line": 153,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "153:15:subscript:subscript:74",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "signs[..., tf.newaxis, :]",
        "line": 153,
        "column": 15,
        "evidence": {}
      },
      {
        "id": "153:42:subscript:subscript:75",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "signs[..., :, tf.newaxis]",
        "line": 153,
        "column": 42,
        "evidence": {}
      },
      {
        "id": "145:47:subscript:subscript:76",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor]",
        "line": 145,
        "column": 47,
        "evidence": {}
      },
      {
        "id": "159:4:assign:assignment:77",
        "kind": "assign",
        "operation": "assignment",
        "target": "_, r",
        "expression": "_batched_qr_positive(_matrix_transpose(stack))",
        "line": 159,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "r"
          ]
        }
      },
      {
        "id": "159:11:call:call:78",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_qr_positive(_matrix_transpose(stack))",
        "line": 159,
        "column": 11,
        "evidence": {
          "function": "_batched_qr_positive"
        }
      },
      {
        "id": "159:32:call:call:79",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(stack)",
        "line": 159,
        "column": 32,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "160:4:return:return:80",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "_matrix_transpose(r)",
        "line": 160,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "160:11:call:call:81",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(r)",
        "line": 160,
        "column": 11,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "166:4:assign:assignment:82",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "common_floating_dtype(factor, rhs, context='batched factor solve inputs')",
        "line": 166,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "166:4:assign:time_step_update:83",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "common_floating_dtype(factor, rhs, context='batched factor solve inputs')",
        "line": 166,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "166:12:call:call:84",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(factor, rhs, context='batched factor solve inputs')",
        "line": 166,
        "column": 12,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "167:4:assign:assignment:85",
        "kind": "assign",
        "operation": "assignment",
        "target": "factor",
        "expression": "as_float_tensor(factor, dtype, name='factor')",
        "line": 167,
        "column": 4,
        "evidence": {
          "targets": [
            "factor"
          ]
        }
      },
      {
        "id": "167:4:assign:time_step_update:86",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "factor",
        "expression": "as_float_tensor(factor, dtype, name='factor')",
        "line": 167,
        "column": 4,
        "evidence": {
          "targets": [
            "factor"
          ]
        }
      },
      {
        "id": "167:13:call:call:87",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "as_float_tensor(factor, dtype, name='factor')",
        "line": 167,
        "column": 13,
        "evidence": {
          "function": "as_float_tensor"
        }
      },
      {
        "id": "168:4:assign:assignment:88",
        "kind": "assign",
        "operation": "assignment",
        "target": "rhs",
        "expression": "as_float_tensor(rhs, dtype, name='rhs')",
        "line": 168,
        "column": 4,
        "evidence": {
          "targets": [
            "rhs"
          ]
        }
      },
      {
        "id": "168:4:assign:innovation_covariance:89",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "rhs",
        "expression": "as_float_tensor(rhs, dtype, name='rhs')",
        "line": 168,
        "column": 4,
        "evidence": {
          "targets": [
            "rhs"
          ]
        }
      },
      {
        "id": "168:4:assign:time_step_update:90",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "rhs",
        "expression": "as_float_tensor(rhs, dtype, name='rhs')",
        "line": 168,
        "column": 4,
        "evidence": {
          "targets": [
            "rhs"
          ]
        }
      },
      {
        "id": "168:10:call:call:91",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "as_float_tensor(rhs, dtype, name='rhs')",
        "line": 168,
        "column": 10,
        "evidence": {
          "function": "as_float_tensor"
        }
      },
      {
        "id": "169:4:assign:assignment:92",
        "kind": "assign",
        "operation": "assignment",
        "target": "first",
        "expression": "tf.linalg.triangular_solve(factor, rhs, lower=True)",
        "line": 169,
        "column": 4,
        "evidence": {
          "targets": [
            "first"
          ]
        }
      },
      {
        "id": "169:4:assign:innovation_covariance:93",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "first",
        "expression": "tf.linalg.triangular_solve(factor, rhs, lower=True)",
        "line": 169,
        "column": 4,
        "evidence": {
          "targets": [
            "first"
          ]
        }
      },
      {
        "id": "169:12:call:call:94",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(factor, rhs, lower=True)",
        "line": 169,
        "column": 12,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "170:4:return:return:95",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf.linalg.triangular_solve(_matrix_transpose(factor), first, lower=False)",
        "line": 170,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "170:11:call:call:96",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(_matrix_transpose(factor), first, lower=False)",
        "line": 170,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "170:38:call:call:97",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(factor)",
        "line": 170,
        "column": 38,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "176:4:assign:assignment:98",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "common_floating_dtype(covariance, jitter, context='batched Cholesky inputs')",
        "line": 176,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "176:4:assign:time_step_update:99",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "common_floating_dtype(covariance, jitter, context='batched Cholesky inputs')",
        "line": 176,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "176:12:call:call:100",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(covariance, jitter, context='batched Cholesky inputs')",
        "line": 176,
        "column": 12,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "177:4:assign:assignment:101",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance",
        "expression": "_to_tensor(covariance, dtype)",
        "line": 177,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance"
          ]
        }
      },
      {
        "id": "177:4:assign:time_step_update:102",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "covariance",
        "expression": "_to_tensor(covariance, dtype)",
        "line": 177,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance"
          ]
        }
      },
      {
        "id": "177:17:call:call:103",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(covariance, dtype)",
        "line": 177,
        "column": 17,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "178:7:compare:shape_guard:104",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "covariance.shape.rank != 3",
        "line": 178,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "179:14:call:call:105",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('batched Cholesky requires covariance shape [batch, dim, dim]')",
        "line": 179,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "180:4:assign:assignment:106",
        "kind": "assign",
        "operation": "assignment",
        "target": "symmetric",
        "expression": "0.5 * (covariance + _matrix_transpose(covariance))",
        "line": 180,
        "column": 4,
        "evidence": {
          "targets": [
            "symmetric"
          ]
        }
      },
      {
        "id": "180:4:assign:innovation_covariance:107",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "symmetric",
        "expression": "0.5 * (covariance + _matrix_transpose(covariance))",
        "line": 180,
        "column": 4,
        "evidence": {
          "targets": [
            "symmetric"
          ]
        }
      },
      {
        "id": "180:36:call:call:108",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(covariance)",
        "line": 180,
        "column": 36,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "181:4:assign:assignment:109",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 181,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "181:4:assign:innovation_covariance:110",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 181,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "181:4:assign:time_step_update:111",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 181,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "181:20:call:call:112",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 181,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "182:4:assign:assignment:113",
        "kind": "assign",
        "operation": "assignment",
        "target": "dim",
        "expression": "tf.shape(symmetric)[-1]",
        "line": 182,
        "column": 4,
        "evidence": {
          "targets": [
            "dim"
          ]
        }
      },
      {
        "id": "182:4:assign:shape_reference:114",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "dim",
        "expression": "tf.shape(symmetric)[-1]",
        "line": 182,
        "column": 4,
        "evidence": {
          "targets": [
            "dim"
          ]
        }
      },
      {
        "id": "182:10:subscript:subscript:115",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(symmetric)[-1]",
        "line": 182,
        "column": 10,
        "evidence": {}
      },
      {
        "id": "182:10:call:call:116",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(symmetric)",
        "line": 182,
        "column": 10,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "183:4:assign:assignment:117",
        "kind": "assign",
        "operation": "assignment",
        "target": "identity",
        "expression": "tf.eye(dim, batch_shape=[tf.shape(symmetric)[0]], dtype=dtype)",
        "line": 183,
        "column": 4,
        "evidence": {
          "targets": [
            "identity"
          ]
        }
      },
      {
        "id": "183:4:assign:shape_reference:118",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "identity",
        "expression": "tf.eye(dim, batch_shape=[tf.shape(symmetric)[0]], dtype=dtype)",
        "line": 183,
        "column": 4,
        "evidence": {
          "targets": [
            "identity"
          ]
        }
      },
      {
        "id": "183:4:assign:time_step_update:119",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "identity",
        "expression": "tf.eye(dim, batch_shape=[tf.shape(symmetric)[0]], dtype=dtype)",
        "line": 183,
        "column": 4,
        "evidence": {
          "targets": [
            "identity"
          ]
        }
      },
      {
        "id": "183:15:call:call:120",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(dim, batch_shape=[tf.shape(symmetric)[0]], dtype=dtype)",
        "line": 183,
        "column": 15,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "183:40:subscript:subscript:121",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(symmetric)[0]",
        "line": 183,
        "column": 40,
        "evidence": {}
      },
      {
        "id": "183:40:call:call:122",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(symmetric)",
        "line": 183,
        "column": 40,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "184:4:return:return:123",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf.linalg.cholesky(symmetric + jitter_tensor * identity)",
        "line": 184,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "184:11:call:call:124",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.cholesky(symmetric + jitter_tensor * identity)",
        "line": 184,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.cholesky"
        }
      },
      {
        "id": "184:11:call:cholesky:125",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "tf.linalg.cholesky(symmetric + jitter_tensor * identity)",
        "line": 184,
        "column": 11,
        "evidence": {
          "function": "tf.linalg.cholesky"
        }
      },
      {
        "id": "203:4:return:return:126",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 203,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "203:11:call:call:127",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 203,
        "column": 11,
        "evidence": {
          "function": "tf_qr_sqrt_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "203:11:call:posterior_or_likelihood:128",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 203,
        "column": 11,
        "evidence": {
          "function": "tf_qr_sqrt_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "214:43:call:call:129",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "bool(jitter_updates_filtered_covariance)",
        "line": 214,
        "column": 43,
        "evidence": {
          "function": "bool"
        }
      },
      {
        "id": "234:4:return:return:130",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, observation_mask=observation_mask, jitter=jitter)",
        "line": 234,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "234:11:call:call:131",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, observation_mask=observation_mask, jitter=jitter)",
        "line": 234,
        "column": 11,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "234:11:call:posterior_or_likelihood:132",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=observations, transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance, observation_mask=observation_mask, jitter=jitter)",
        "line": 234,
        "column": 11,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "265:4:assign:assignment:133",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR compact value inputs')",
        "line": 265,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "265:4:assign:time_step_update:134",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR compact value inputs')",
        "line": 265,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "265:12:call:call:135",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR compact value inputs')",
        "line": 265,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "278:4:assign:assignment:136",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 278,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "278:4:assign:time_step_update:137",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 278,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "278:8:call:call:138",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 278,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "279:4:assign:assignment:139",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 279,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "279:4:assign:reparameterization_gradient:140",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 279,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "279:18:call:call:141",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 279,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "280:4:assign:assignment:142",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 280,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "280:4:assign:innovation_covariance:143",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 280,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "280:4:assign:time_step_update:144",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 280,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "280:24:call:call:145",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 280,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "281:4:assign:assignment:146",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 281,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "281:4:assign:innovation_covariance:147",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 281,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "281:4:assign:time_step_update:148",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 281,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "281:24:call:call:149",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 281,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "282:4:assign:assignment:150",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 282,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "282:4:assign:innovation_covariance:151",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 282,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "282:4:assign:time_step_update:152",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 282,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "282:28:call:call:153",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 282,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "283:4:assign:assignment:154",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 283,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "283:4:assign:innovation_covariance:155",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 283,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "283:4:assign:time_step_update:156",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 283,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "283:25:call:call:157",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 283,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "284:4:assign:assignment:158",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 284,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "284:4:assign:innovation_covariance:159",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 284,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "284:4:assign:time_step_update:160",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 284,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "284:25:call:call:161",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 284,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "285:4:assign:assignment:162",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 285,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "285:4:assign:innovation_covariance:163",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 285,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "285:4:assign:time_step_update:164",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 285,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "285:29:call:call:165",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 285,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "286:4:assign:assignment:166",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 286,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "286:4:assign:time_step_update:167",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 286,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "286:11:call:call:168",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 286,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "287:4:assign:assignment:169",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 287,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "287:4:assign:innovation_covariance:170",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 287,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "287:4:assign:time_step_update:171",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 287,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "287:31:call:call:172",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 287,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "288:4:assign:assignment:173",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 288,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "288:4:assign:innovation_covariance:174",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 288,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "288:4:assign:time_step_update:175",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 288,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "288:20:call:call:176",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 288,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "290:4:assign:assignment:177",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 290,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "290:4:assign:innovation_covariance:178",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 290,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "290:4:assign:shape_reference:179",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 290,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "290:16:subscript:subscript:180",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 290,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "290:16:call:call:181",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 290,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "291:4:assign:assignment:182",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 291,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "291:4:assign:innovation_covariance:183",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 291,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "291:4:assign:shape_reference:184",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 291,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "291:4:assign:time_step_update:185",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 291,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "291:14:subscript:subscript:186",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 291,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "291:14:call:call:187",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))",
        "line": 291,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "291:23:call:call:188",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32))",
        "line": 291,
        "column": 23,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "291:59:call:call:189",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 291,
        "column": 59,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "292:4:assign:assignment:190",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 292,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "292:4:assign:time_step_update:191",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 292,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "292:21:call:call:192",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 292,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "293:4:assign:assignment:193",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 293,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "293:4:assign:innovation_covariance:194",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 293,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "293:4:assign:time_step_update:195",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 293,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "293:19:call:call:196",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 293,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "294:4:assign:assignment:197",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 294,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "294:24:call:call:198",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 294,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "294:24:call:cholesky:199",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 294,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "295:4:assign:assignment:200",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 295,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "295:4:assign:kalman_gain:201",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 295,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "295:4:assign:posterior_or_likelihood:202",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 295,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "295:4:assign:time_step_update:203",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 295,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "295:21:call:call:204",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 295,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "296:4:assign:assignment:205",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 296,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "296:4:assign:time_step_update:206",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 296,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "296:13:call:call:207",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 296,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "298:4:loop:loop:208",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 298,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "298:13:call:call:209",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 298,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "299:8:assign:assignment:210",
        "kind": "assign",
        "operation": "assignment",
        "target": "c",
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 299,
        "column": 8,
        "evidence": {
          "targets": [
            "c"
          ]
        }
      },
      {
        "id": "299:12:call:call:211",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 299,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "300:8:assign:assignment:212",
        "kind": "assign",
        "operation": "assignment",
        "target": "T",
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 300,
        "column": 8,
        "evidence": {
          "targets": [
            "T"
          ]
        }
      },
      {
        "id": "300:12:call:call:213",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 300,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "301:8:assign:assignment:214",
        "kind": "assign",
        "operation": "assignment",
        "target": "Q",
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 301,
        "column": 8,
        "evidence": {
          "targets": [
            "Q"
          ]
        }
      },
      {
        "id": "301:12:call:call:215",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 301,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "302:8:assign:assignment:216",
        "kind": "assign",
        "operation": "assignment",
        "target": "d",
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 302,
        "column": 8,
        "evidence": {
          "targets": [
            "d"
          ]
        }
      },
      {
        "id": "302:12:call:call:217",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 302,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "303:8:assign:assignment:218",
        "kind": "assign",
        "operation": "assignment",
        "target": "Z",
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 303,
        "column": 8,
        "evidence": {
          "targets": [
            "Z"
          ]
        }
      },
      {
        "id": "303:12:call:call:219",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 303,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "304:8:assign:assignment:220",
        "kind": "assign",
        "operation": "assignment",
        "target": "H",
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 304,
        "column": 8,
        "evidence": {
          "targets": [
            "H"
          ]
        }
      },
      {
        "id": "304:12:call:call:221",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 304,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "305:8:assign:assignment:222",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 305,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "305:8:assign:innovation_covariance:223",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 305,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "305:39:call:call:224",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 305,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "305:39:call:cholesky:225",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 305,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "306:8:assign:assignment:226",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 306,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "306:8:assign:innovation_covariance:227",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 306,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "306:40:call:call:228",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 306,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "306:40:call:cholesky:229",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 306,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "307:8:assign:assignment:230",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 307,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "307:8:assign:innovation_covariance:231",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 307,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "310:17:call:call:232",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 310,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "310:17:call:cholesky:233",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 310,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "313:8:assign:assignment:234",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 313,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "313:8:assign:prediction_update:235",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 313,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "313:29:call:call:236",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(T, mean)",
        "line": 313,
        "column": 29,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "314:8:assign:assignment:237",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:8:assign:innovation_covariance:238",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:8:assign:kalman_gain:239",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:8:assign:matmul:240",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:8:assign:prediction_update:241",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:8:assign:quadratic_form:242",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "314:27:call:call:243",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 314,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "315:13:binop:matmul:244",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 315,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "315:13:binop:quadratic_form:245",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 315,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "318:8:assign:assignment:246",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 318,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "318:27:call:call:247",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 318,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "319:8:assign:assignment:248",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "319:8:assign:matmul:249",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "319:8:assign:prediction_update:250",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "319:8:assign:quadratic_form:251",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "319:31:binop:matmul:252",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "319:31:binop:quadratic_form:253",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 319,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "319:50:call:call:254",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 319,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "321:8:assign:assignment:255",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 321,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "321:8:assign:innovation_update:256",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 321,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "321:8:assign:prediction_update:257",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 321,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "321:21:subscript:subscript:258",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 321,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "321:33:call:call:259",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(Z, predicted_mean)",
        "line": 321,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "322:8:assign:assignment:260",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:innovation_covariance:261",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:innovation_update:262",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:kalman_gain:263",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:matmul:264",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:prediction_update:265",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:8:assign:quadratic_form:266",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "322:27:call:call:267",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 322,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "323:13:binop:matmul:268",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 323,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "323:13:binop:quadratic_form:269",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 323,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "326:8:assign:assignment:270",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 326,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "326:8:assign:innovation_update:271",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 326,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "326:28:call:call:272",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 326,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "327:8:assign:assignment:273",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 327,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "327:8:assign:innovation_covariance:274",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 327,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "327:8:assign:innovation_update:275",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 327,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "327:31:call:call:276",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 327,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "328:8:assign:assignment:277",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:8:assign:innovation_update:278",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:8:assign:kalman_gain:279",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:8:assign:matmul:280",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:8:assign:prediction_update:281",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:8:assign:quadratic_form:282",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "328:22:binop:matmul:283",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "328:22:binop:quadratic_form:284",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 328,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "328:22:binop:matmul:285",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 328,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "328:22:binop:quadratic_form:286",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 328,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "328:45:call:call:287",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(Z)",
        "line": 328,
        "column": 45,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "330:8:assign:assignment:288",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "330:8:assign:innovation_update:289",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "330:8:assign:kalman_gain:290",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "330:8:assign:prediction_update:291",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "330:8:assign:state_update:292",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "330:41:call:call:293",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 330,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "331:8:assign:assignment:294",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:8:assign:covariance_update:295",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:8:assign:innovation_covariance:296",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:8:assign:kalman_gain:297",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:8:assign:matmul:298",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:8:assign:quadratic_form:299",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 331,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "331:39:binop:matmul:300",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 331,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "331:39:binop:quadratic_form:301",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 331,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "332:8:assign:assignment:302",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:covariance_update:303",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:innovation_covariance:304",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:kalman_gain:305",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:matmul:306",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:prediction_update:307",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:8:assign:quadratic_form:308",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "332:23:call:call:309",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 332,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "334:16:binop:matmul:310",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 334,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "334:16:binop:quadratic_form:311",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 334,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "335:16:binop:matmul:312",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 335,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "335:16:binop:quadratic_form:313",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 335,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "339:8:assign:assignment:314",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 339,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "339:26:call:call:315",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 339,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "341:8:assign:assignment:316",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 341,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "341:8:assign:innovation_covariance:317",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 341,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "341:8:assign:innovation_update:318",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 341,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "341:27:call:call:319",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 341,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "343:12:subscript:subscript:320",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 343,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "346:8:assign:assignment:321",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 346,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "346:8:assign:innovation_covariance:322",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 346,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "346:8:assign:innovation_update:323",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 346,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "346:22:call:call:324",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 346,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "346:36:call:call:325",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 346,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "347:8:assign:assignment:326",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 347,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "347:8:assign:innovation_update:327",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 347,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "347:24:call:call:328",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 347,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "347:38:call:call:329",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 347,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "347:50:call:call:330",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 347,
        "column": 50,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "348:8:assign:assignment:331",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 348,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "348:8:assign:time_step_update:332",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 348,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "349:12:call:call:333",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 349,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "349:38:call:call:334",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 349,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "351:8:assign:assignment:335",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 351,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "352:8:assign:assignment:336",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 352,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "353:8:assign:assignment:337",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 353,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "353:8:assign:kalman_gain:338",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 353,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "353:8:assign:posterior_or_likelihood:339",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 353,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "355:4:return:return:340",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 355,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "381:4:assign:assignment:341",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR while-loop value inputs')",
        "line": 381,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "381:4:assign:time_step_update:342",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR while-loop value inputs')",
        "line": 381,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "381:12:call:call:343",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR while-loop value inputs')",
        "line": 381,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "394:4:assign:assignment:344",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 394,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "394:4:assign:time_step_update:345",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 394,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "394:8:call:call:346",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 394,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "395:4:assign:assignment:347",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 395,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "395:4:assign:innovation_covariance:348",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 395,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "395:4:assign:reparameterization_gradient:349",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 395,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "395:4:assign:shape_reference:350",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 395,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "395:18:subscript:subscript:351",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(y)[0]",
        "line": 395,
        "column": 18,
        "evidence": {}
      },
      {
        "id": "395:18:call:call:352",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(y)",
        "line": 395,
        "column": 18,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "396:4:assign:assignment:353",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 396,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "396:4:assign:innovation_covariance:354",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 396,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "396:4:assign:time_step_update:355",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 396,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "396:24:call:call:356",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 396,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "397:4:assign:assignment:357",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 397,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "397:4:assign:innovation_covariance:358",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 397,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "397:4:assign:time_step_update:359",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 397,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "397:24:call:call:360",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 397,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "398:4:assign:assignment:361",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 398,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "398:4:assign:innovation_covariance:362",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 398,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "398:4:assign:time_step_update:363",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 398,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "398:28:call:call:364",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 398,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "399:4:assign:assignment:365",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 399,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "399:4:assign:innovation_covariance:366",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 399,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "399:4:assign:time_step_update:367",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 399,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "399:25:call:call:368",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 399,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "400:4:assign:assignment:369",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 400,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "400:4:assign:innovation_covariance:370",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 400,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "400:4:assign:time_step_update:371",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 400,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "400:25:call:call:372",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 400,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "401:4:assign:assignment:373",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 401,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "401:4:assign:innovation_covariance:374",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 401,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "401:4:assign:time_step_update:375",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 401,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "401:29:call:call:376",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 401,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "402:4:assign:assignment:377",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 402,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "402:4:assign:time_step_update:378",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 402,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "402:12:call:call:379",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 402,
        "column": 12,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "403:4:assign:assignment:380",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 403,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "403:4:assign:innovation_covariance:381",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 403,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "403:4:assign:time_step_update:382",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 403,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "403:31:call:call:383",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 403,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "404:4:assign:assignment:384",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 404,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "404:4:assign:innovation_covariance:385",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 404,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "404:4:assign:time_step_update:386",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 404,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "404:20:call:call:387",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 404,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "406:4:assign:assignment:388",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 406,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "406:4:assign:innovation_covariance:389",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 406,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "406:4:assign:shape_reference:390",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 406,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "406:16:subscript:subscript:391",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean0)[0]",
        "line": 406,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "406:16:call:call:392",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean0)",
        "line": 406,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "407:4:assign:assignment:393",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 407,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "407:4:assign:innovation_covariance:394",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 407,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "407:4:assign:shape_reference:395",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 407,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "407:4:assign:time_step_update:396",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 407,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "407:14:subscript:subscript:397",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 407,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "407:14:call:call:398",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))",
        "line": 407,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "407:23:call:call:399",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32))",
        "line": 407,
        "column": 23,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "407:59:call:call:400",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 407,
        "column": 59,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "408:4:assign:assignment:401",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 408,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "408:4:assign:time_step_update:402",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 408,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "408:21:call:call:403",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 408,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "409:4:assign:assignment:404",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 409,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "409:4:assign:innovation_covariance:405",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 409,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "409:4:assign:time_step_update:406",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 409,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "409:19:call:call:407",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 409,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "410:4:assign:assignment:408",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor0",
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 410,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor0"
          ]
        }
      },
      {
        "id": "410:25:call:call:409",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 410,
        "column": 25,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "410:25:call:cholesky:410",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 410,
        "column": 25,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "411:4:assign:assignment:411",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 411,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "411:4:assign:kalman_gain:412",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 411,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "411:4:assign:posterior_or_likelihood:413",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 411,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "411:4:assign:time_step_update:414",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 411,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "411:22:call:call:415",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 411,
        "column": 22,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "412:4:assign:assignment:416",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 412,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "412:4:assign:time_step_update:417",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 412,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "412:13:call:call:418",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 412,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "413:4:assign:assignment:419",
        "kind": "assign",
        "operation": "assignment",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 413,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "413:4:assign:time_step_update:420",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 413,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "413:9:call:call:421",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 413,
        "column": 9,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "416:8:return:return:422",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "t < n_timesteps",
        "line": 416,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "419:8:assign:assignment:423",
        "kind": "assign",
        "operation": "assignment",
        "target": "c",
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 419,
        "column": 8,
        "evidence": {
          "targets": [
            "c"
          ]
        }
      },
      {
        "id": "419:12:call:call:424",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 419,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "420:8:assign:assignment:425",
        "kind": "assign",
        "operation": "assignment",
        "target": "T",
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 420,
        "column": 8,
        "evidence": {
          "targets": [
            "T"
          ]
        }
      },
      {
        "id": "420:12:call:call:426",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 420,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "421:8:assign:assignment:427",
        "kind": "assign",
        "operation": "assignment",
        "target": "Q",
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 421,
        "column": 8,
        "evidence": {
          "targets": [
            "Q"
          ]
        }
      },
      {
        "id": "421:12:call:call:428",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 421,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "422:8:assign:assignment:429",
        "kind": "assign",
        "operation": "assignment",
        "target": "d",
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 422,
        "column": 8,
        "evidence": {
          "targets": [
            "d"
          ]
        }
      },
      {
        "id": "422:12:call:call:430",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 422,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "423:8:assign:assignment:431",
        "kind": "assign",
        "operation": "assignment",
        "target": "Z",
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 423,
        "column": 8,
        "evidence": {
          "targets": [
            "Z"
          ]
        }
      },
      {
        "id": "423:12:call:call:432",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 423,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "424:8:assign:assignment:433",
        "kind": "assign",
        "operation": "assignment",
        "target": "H",
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 424,
        "column": 8,
        "evidence": {
          "targets": [
            "H"
          ]
        }
      },
      {
        "id": "424:12:call:call:434",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 424,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "425:8:assign:assignment:435",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 425,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "425:8:assign:innovation_covariance:436",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 425,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "425:39:call:call:437",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 425,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "425:39:call:cholesky:438",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 425,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "426:8:assign:assignment:439",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 426,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "426:8:assign:innovation_covariance:440",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 426,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "426:40:call:call:441",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 426,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "426:40:call:cholesky:442",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 426,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "427:8:assign:assignment:443",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 427,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "427:8:assign:innovation_covariance:444",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 427,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "430:17:call:call:445",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 430,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "430:17:call:cholesky:446",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 430,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "433:8:assign:assignment:447",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 433,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "433:8:assign:prediction_update:448",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 433,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "433:29:call:call:449",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(T, mean)",
        "line": 433,
        "column": 29,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "434:8:assign:assignment:450",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:8:assign:innovation_covariance:451",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:8:assign:kalman_gain:452",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:8:assign:matmul:453",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:8:assign:prediction_update:454",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:8:assign:quadratic_form:455",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "434:27:call:call:456",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 434,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "435:13:binop:matmul:457",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 435,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "435:13:binop:quadratic_form:458",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 435,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "438:8:assign:assignment:459",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 438,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "438:27:call:call:460",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 438,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "439:8:assign:assignment:461",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "439:8:assign:matmul:462",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "439:8:assign:prediction_update:463",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "439:8:assign:quadratic_form:464",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "439:31:binop:matmul:465",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "439:31:binop:quadratic_form:466",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 439,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "439:50:call:call:467",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 439,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "441:8:assign:assignment:468",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 441,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "441:8:assign:innovation_update:469",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 441,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "441:8:assign:prediction_update:470",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 441,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "441:21:subscript:subscript:471",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 441,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "441:33:call:call:472",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(Z, predicted_mean)",
        "line": 441,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "442:8:assign:assignment:473",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:innovation_covariance:474",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:innovation_update:475",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:kalman_gain:476",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:matmul:477",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:prediction_update:478",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:8:assign:quadratic_form:479",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "442:27:call:call:480",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 442,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "443:13:binop:matmul:481",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 443,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "443:13:binop:quadratic_form:482",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 443,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "446:8:assign:assignment:483",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 446,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "446:8:assign:innovation_update:484",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 446,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "446:28:call:call:485",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 446,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "447:8:assign:assignment:486",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 447,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "447:8:assign:innovation_covariance:487",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 447,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "447:8:assign:innovation_update:488",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 447,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "447:31:call:call:489",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 447,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "448:8:assign:assignment:490",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:8:assign:innovation_update:491",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:8:assign:kalman_gain:492",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:8:assign:matmul:493",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:8:assign:prediction_update:494",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:8:assign:quadratic_form:495",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "448:22:binop:matmul:496",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "448:22:binop:quadratic_form:497",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 448,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "448:22:binop:matmul:498",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 448,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "448:22:binop:quadratic_form:499",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 448,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "448:45:call:call:500",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(Z)",
        "line": 448,
        "column": 45,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "450:8:assign:assignment:501",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "450:8:assign:innovation_update:502",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "450:8:assign:kalman_gain:503",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "450:8:assign:prediction_update:504",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "450:8:assign:state_update:505",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "450:41:call:call:506",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 450,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "451:8:assign:assignment:507",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:8:assign:covariance_update:508",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:8:assign:innovation_covariance:509",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:8:assign:kalman_gain:510",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:8:assign:matmul:511",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:8:assign:quadratic_form:512",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 451,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "451:39:binop:matmul:513",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 451,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "451:39:binop:quadratic_form:514",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 451,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "452:8:assign:assignment:515",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:covariance_update:516",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:innovation_covariance:517",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:kalman_gain:518",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:matmul:519",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:prediction_update:520",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:8:assign:quadratic_form:521",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "452:23:call:call:522",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 452,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "454:16:binop:matmul:523",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 454,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "454:16:binop:quadratic_form:524",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 454,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "455:16:binop:matmul:525",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 455,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "455:16:binop:quadratic_form:526",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 455,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "459:8:assign:assignment:527",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 459,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "459:26:call:call:528",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 459,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "461:8:assign:assignment:529",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 461,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "461:8:assign:innovation_covariance:530",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 461,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "461:8:assign:innovation_update:531",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 461,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "461:27:call:call:532",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 461,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "463:12:subscript:subscript:533",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 463,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "466:8:assign:assignment:534",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 466,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "466:8:assign:innovation_covariance:535",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 466,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "466:8:assign:innovation_update:536",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 466,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "466:22:call:call:537",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 466,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "466:36:call:call:538",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 466,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "467:8:assign:assignment:539",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 467,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "467:8:assign:innovation_update:540",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 467,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "467:24:call:call:541",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 467,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "467:38:call:call:542",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 467,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "467:50:call:call:543",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 467,
        "column": 50,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "468:8:assign:assignment:544",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 468,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "468:8:assign:time_step_update:545",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 468,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "469:12:call:call:546",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 469,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "469:38:call:call:547",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 469,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "471:8:return:return:548",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(t + 1, filtered_mean, filtered_factor, log_likelihood + contribution)",
        "line": 471,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "473:4:assign:assignment:549",
        "kind": "assign",
        "operation": "assignment",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 473,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "473:4:assign:kalman_gain:550",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 473,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "473:4:assign:posterior_or_likelihood:551",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 473,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "473:4:assign:reparameterization_gradient:552",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 473,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "473:30:call:call:553",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 473,
        "column": 30,
        "evidence": {
          "function": "tf.while_loop"
        }
      },
      {
        "id": "480:4:return:return:554",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 480,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "358:1:call:call:555",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.function(reduce_retracing=True)",
        "line": 358,
        "column": 1,
        "evidence": {
          "function": "tf.function"
        }
      },
      {
        "id": "498:4:assign:assignment:556",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, context='factorized QR likelihood inputs')",
        "line": 498,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "498:4:assign:posterior_or_likelihood:557",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "dtype",
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, context='factorized QR likelihood inputs')",
        "line": 498,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "498:4:assign:time_step_update:558",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, context='factorized QR likelihood inputs')",
        "line": 498,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "498:12:call:call:559",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "common_floating_dtype(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, context='factorized QR likelihood inputs')",
        "line": 498,
        "column": 12,
        "evidence": {
          "function": "common_floating_dtype"
        }
      },
      {
        "id": "511:4:assign:assignment:560",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 511,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "511:4:assign:time_step_update:561",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 511,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "511:8:call:call:562",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 511,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "512:4:assign:assignment:563",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 512,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "512:4:assign:innovation_covariance:564",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 512,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "512:4:assign:time_step_update:565",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 512,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "512:24:call:call:566",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 512,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "513:4:assign:assignment:567",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 513,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "513:4:assign:innovation_covariance:568",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 513,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "513:4:assign:time_step_update:569",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 513,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "513:24:call:call:570",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 513,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "514:4:assign:assignment:571",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_noise_factor",
        "expression": "_to_tensor(transition_noise_factor, dtype)",
        "line": 514,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_noise_factor"
          ]
        }
      },
      {
        "id": "514:4:assign:innovation_covariance:572",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_noise_factor",
        "expression": "_to_tensor(transition_noise_factor, dtype)",
        "line": 514,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_noise_factor"
          ]
        }
      },
      {
        "id": "514:4:assign:time_step_update:573",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_noise_factor",
        "expression": "_to_tensor(transition_noise_factor, dtype)",
        "line": 514,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_noise_factor"
          ]
        }
      },
      {
        "id": "514:30:call:call:574",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_noise_factor, dtype)",
        "line": 514,
        "column": 30,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "515:4:assign:assignment:575",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 515,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "515:4:assign:innovation_covariance:576",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 515,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "515:4:assign:time_step_update:577",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 515,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "515:25:call:call:578",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 515,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "516:4:assign:assignment:579",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 516,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "516:4:assign:innovation_covariance:580",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 516,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "516:4:assign:time_step_update:581",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 516,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "516:25:call:call:582",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 516,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "517:4:assign:assignment:583",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 517,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "517:4:assign:innovation_covariance:584",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 517,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "517:4:assign:time_step_update:585",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 517,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "517:29:call:call:586",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 517,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "518:4:assign:assignment:587",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 518,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "518:4:assign:time_step_update:588",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 518,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "518:12:call:call:589",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 518,
        "column": 12,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "519:4:assign:assignment:590",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_factor",
        "expression": "_to_tensor(initial_state_factor, dtype)",
        "line": 519,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_factor"
          ]
        }
      },
      {
        "id": "519:4:assign:innovation_covariance:591",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_factor",
        "expression": "_to_tensor(initial_state_factor, dtype)",
        "line": 519,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_factor"
          ]
        }
      },
      {
        "id": "519:4:assign:time_step_update:592",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_factor",
        "expression": "_to_tensor(initial_state_factor, dtype)",
        "line": 519,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_factor"
          ]
        }
      },
      {
        "id": "519:27:call:call:593",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_factor, dtype)",
        "line": 519,
        "column": 27,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "520:4:assign:assignment:594",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 520,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "520:4:assign:innovation_covariance:595",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 520,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "520:4:assign:time_step_update:596",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 520,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "520:20:call:call:597",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 520,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "522:7:compare:shape_guard:598",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "transition_noise_factor.shape.rank != 2",
        "line": 522,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "523:14:call:call:599",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('transition_noise_factor must have rank 2')",
        "line": 523,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "524:7:compare:shape_guard:600",
        "kind": "compare",
        "operation": "shape_guard",
        "target": null,
        "expression": "initial_state_factor.shape.rank != 2",
        "line": 524,
        "column": 7,
        "evidence": {}
      },
      {
        "id": "525:14:call:call:601",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('initial_state_factor must have rank 2')",
        "line": 525,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "526:4:assign:assignment:602",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim_static",
        "expression": "mean0.shape[0]",
        "line": 526,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim_static"
          ]
        }
      },
      {
        "id": "526:4:assign:innovation_covariance:603",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim_static",
        "expression": "mean0.shape[0]",
        "line": 526,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim_static"
          ]
        }
      },
      {
        "id": "526:4:assign:shape_reference:604",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim_static",
        "expression": "mean0.shape[0]",
        "line": 526,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim_static"
          ]
        }
      },
      {
        "id": "526:23:subscript:subscript:605",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "mean0.shape[0]",
        "line": 526,
        "column": 23,
        "evidence": {}
      },
      {
        "id": "527:4:assign:assignment:606",
        "kind": "assign",
        "operation": "assignment",
        "target": "factor_state_dim",
        "expression": "transition_noise_factor.shape[0]",
        "line": 527,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_state_dim"
          ]
        }
      },
      {
        "id": "527:4:assign:innovation_covariance:607",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "factor_state_dim",
        "expression": "transition_noise_factor.shape[0]",
        "line": 527,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_state_dim"
          ]
        }
      },
      {
        "id": "527:4:assign:shape_reference:608",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "factor_state_dim",
        "expression": "transition_noise_factor.shape[0]",
        "line": 527,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_state_dim"
          ]
        }
      },
      {
        "id": "527:23:subscript:subscript:609",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "transition_noise_factor.shape[0]",
        "line": 527,
        "column": 23,
        "evidence": {}
      },
      {
        "id": "528:4:assign:assignment:610",
        "kind": "assign",
        "operation": "assignment",
        "target": "factor_innovation_dim",
        "expression": "transition_noise_factor.shape[1]",
        "line": 528,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_innovation_dim"
          ]
        }
      },
      {
        "id": "528:4:assign:innovation_update:611",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "factor_innovation_dim",
        "expression": "transition_noise_factor.shape[1]",
        "line": 528,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_innovation_dim"
          ]
        }
      },
      {
        "id": "528:4:assign:shape_reference:612",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "factor_innovation_dim",
        "expression": "transition_noise_factor.shape[1]",
        "line": 528,
        "column": 4,
        "evidence": {
          "targets": [
            "factor_innovation_dim"
          ]
        }
      },
      {
        "id": "528:28:subscript:subscript:613",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "transition_noise_factor.shape[1]",
        "line": 528,
        "column": 28,
        "evidence": {}
      },
      {
        "id": "529:4:assign:assignment:614",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_factor_state_dim",
        "expression": "initial_state_factor.shape[0]",
        "line": 529,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_state_dim"
          ]
        }
      },
      {
        "id": "529:4:assign:innovation_covariance:615",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_factor_state_dim",
        "expression": "initial_state_factor.shape[0]",
        "line": 529,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_state_dim"
          ]
        }
      },
      {
        "id": "529:4:assign:shape_reference:616",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "initial_factor_state_dim",
        "expression": "initial_state_factor.shape[0]",
        "line": 529,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_state_dim"
          ]
        }
      },
      {
        "id": "529:31:subscript:subscript:617",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "initial_state_factor.shape[0]",
        "line": 529,
        "column": 31,
        "evidence": {}
      },
      {
        "id": "530:4:assign:assignment:618",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_factor_width",
        "expression": "initial_state_factor.shape[1]",
        "line": 530,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_width"
          ]
        }
      },
      {
        "id": "530:4:assign:shape_reference:619",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "initial_factor_width",
        "expression": "initial_state_factor.shape[1]",
        "line": 530,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_width"
          ]
        }
      },
      {
        "id": "530:4:assign:time_step_update:620",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_factor_width",
        "expression": "initial_state_factor.shape[1]",
        "line": 530,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_factor_width"
          ]
        }
      },
      {
        "id": "530:27:subscript:subscript:621",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "initial_state_factor.shape[1]",
        "line": 530,
        "column": 27,
        "evidence": {}
      },
      {
        "id": "538:14:call:call:622",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('initial state and covariance factors need static dimensions')",
        "line": 538,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "542:14:call:call:623",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('transition_noise_factor first dimension must equal state dimension')",
        "line": 542,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "546:14:call:call:624",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('initial_state_factor first dimension must equal state dimension')",
        "line": 546,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "550:4:assign:assignment:625",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 550,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "550:4:assign:innovation_covariance:626",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 550,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "550:4:assign:reparameterization_gradient:627",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 550,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "550:4:assign:shape_reference:628",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 550,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "550:18:subscript:subscript:629",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(y)[0]",
        "line": 550,
        "column": 18,
        "evidence": {}
      },
      {
        "id": "550:18:call:call:630",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(y)",
        "line": 550,
        "column": 18,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "551:4:assign:assignment:631",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 551,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "551:4:assign:innovation_covariance:632",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 551,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "551:4:assign:shape_reference:633",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[0]",
        "line": 551,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "551:16:subscript:subscript:634",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean0)[0]",
        "line": 551,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "551:16:call:call:635",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean0)",
        "line": 551,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "552:4:assign:assignment:636",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(observation_matrix)[0]",
        "line": 552,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "552:4:assign:innovation_covariance:637",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(observation_matrix)[0]",
        "line": 552,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "552:4:assign:shape_reference:638",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(observation_matrix)[0]",
        "line": 552,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "552:14:subscript:subscript:639",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(observation_matrix)[0]",
        "line": 552,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "552:14:call:call:640",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observation_matrix)",
        "line": 552,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "553:4:assign:assignment:641",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 553,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "553:4:assign:time_step_update:642",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 553,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "553:21:call:call:643",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 553,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "554:4:assign:assignment:644",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 554,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "554:4:assign:innovation_covariance:645",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 554,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "554:4:assign:time_step_update:646",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 554,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "554:19:call:call:647",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 554,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "555:4:assign:assignment:648",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor0",
        "expression": "lower_factor_from_horizontal_stack(initial_state_factor)",
        "line": 555,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor0"
          ]
        }
      },
      {
        "id": "555:25:call:call:649",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(initial_state_factor)",
        "line": 555,
        "column": 25,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "556:4:assign:assignment:650",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 556,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "556:4:assign:innovation_covariance:651",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 556,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "556:36:call:call:652",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 556,
        "column": 36,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "556:36:call:cholesky:653",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 556,
        "column": 36,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "560:4:assign:assignment:654",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(observation_covariance, 0.0)",
        "line": 560,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "560:4:assign:innovation_covariance:655",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(observation_covariance, 0.0)",
        "line": 560,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "563:13:call:call:656",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(observation_covariance, 0.0)",
        "line": 563,
        "column": 13,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "563:13:call:cholesky:657",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(observation_covariance, 0.0)",
        "line": 563,
        "column": 13,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "565:4:assign:assignment:658",
        "kind": "assign",
        "operation": "assignment",
        "target": "increments0",
        "expression": "tf.zeros((tf.shape(y)[0],), dtype=dtype)",
        "line": 565,
        "column": 4,
        "evidence": {
          "targets": [
            "increments0"
          ]
        }
      },
      {
        "id": "565:4:assign:innovation_covariance:659",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "increments0",
        "expression": "tf.zeros((tf.shape(y)[0],), dtype=dtype)",
        "line": 565,
        "column": 4,
        "evidence": {
          "targets": [
            "increments0"
          ]
        }
      },
      {
        "id": "565:4:assign:shape_reference:660",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "increments0",
        "expression": "tf.zeros((tf.shape(y)[0],), dtype=dtype)",
        "line": 565,
        "column": 4,
        "evidence": {
          "targets": [
            "increments0"
          ]
        }
      },
      {
        "id": "565:4:assign:time_step_update:661",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "increments0",
        "expression": "tf.zeros((tf.shape(y)[0],), dtype=dtype)",
        "line": 565,
        "column": 4,
        "evidence": {
          "targets": [
            "increments0"
          ]
        }
      },
      {
        "id": "565:18:call:call:662",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.zeros((tf.shape(y)[0],), dtype=dtype)",
        "line": 565,
        "column": 18,
        "evidence": {
          "function": "tf.zeros"
        }
      },
      {
        "id": "565:28:subscript:subscript:663",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(y)[0]",
        "line": 565,
        "column": 28,
        "evidence": {}
      },
      {
        "id": "565:28:call:call:664",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(y)",
        "line": 565,
        "column": 28,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "566:4:assign:assignment:665",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 566,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "566:4:assign:time_step_update:666",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 566,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "566:13:call:call:667",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 566,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "567:4:assign:assignment:668",
        "kind": "assign",
        "operation": "assignment",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 567,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "567:4:assign:time_step_update:669",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 567,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "567:9:call:call:670",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 567,
        "column": 9,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "568:4:assign:assignment:671",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 568,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "568:4:assign:kalman_gain:672",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 568,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "568:4:assign:posterior_or_likelihood:673",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 568,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "568:4:assign:time_step_update:674",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood0",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 568,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "568:22:call:call:675",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 568,
        "column": 22,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "571:8:return:return:676",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "t < n_timesteps",
        "line": 571,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "574:8:assign:assignment:677",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 574,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "574:8:assign:prediction_update:678",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 574,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "574:45:call:call:679",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(transition_matrix, mean)",
        "line": 574,
        "column": 45,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "575:8:assign:assignment:680",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1))",
        "line": 575,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "575:8:assign:matmul:681",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1))",
        "line": 575,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "575:8:assign:prediction_update:682",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1))",
        "line": 575,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "575:8:assign:quadratic_form:683",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1))",
        "line": 575,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "575:27:call:call:684",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1))",
        "line": 575,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "576:12:call:call:685",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_noise_factor), axis=1)",
        "line": 576,
        "column": 12,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "578:20:binop:matmul:686",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 578,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "578:20:binop:quadratic_form:687",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 578,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "584:8:assign:assignment:688",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "584:8:assign:matmul:689",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "584:8:assign:prediction_update:690",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "584:8:assign:quadratic_form:691",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "584:31:binop:matmul:692",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "584:31:binop:quadratic_form:693",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 584,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "584:50:call:call:694",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 584,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "586:8:assign:assignment:695",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 586,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "586:8:assign:innovation_update:696",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 586,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "586:8:assign:prediction_update:697",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 586,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "586:21:subscript:subscript:698",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 586,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "587:33:call:call:699",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(observation_matrix, predicted_mean)",
        "line": 587,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "589:8:assign:assignment:700",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "589:8:assign:innovation_update:701",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "589:8:assign:matmul:702",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "589:8:assign:prediction_update:703",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "589:8:assign:quadratic_form:704",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "589:28:call:call:705",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1))",
        "line": 589,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "590:12:call:call:706",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 590,
        "column": 12,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "592:20:binop:matmul:707",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 592,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "592:20:binop:quadratic_form:708",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 592,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "598:8:assign:assignment:709",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 598,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "598:8:assign:innovation_covariance:710",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 598,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "598:8:assign:innovation_update:711",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 598,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "598:31:call:call:712",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 598,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "599:8:assign:assignment:713",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "599:8:assign:innovation_update:714",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "599:8:assign:kalman_gain:715",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "599:8:assign:matmul:716",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "599:8:assign:prediction_update:717",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "599:8:assign:quadratic_form:718",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 599,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "600:12:binop:matmul:719",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 600,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "600:12:binop:quadratic_form:720",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision",
        "line": 600,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "600:12:binop:matmul:721",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(observation_matrix)",
        "line": 600,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "600:12:binop:quadratic_form:722",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(observation_matrix)",
        "line": 600,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "601:14:call:call:723",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(observation_matrix)",
        "line": 601,
        "column": 14,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "605:8:assign:assignment:724",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "605:8:assign:innovation_update:725",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "605:8:assign:kalman_gain:726",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "605:8:assign:prediction_update:727",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "605:8:assign:state_update:728",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "605:41:call:call:729",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 605,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "606:8:assign:assignment:730",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:8:assign:covariance_update:731",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:8:assign:innovation_covariance:732",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:8:assign:kalman_gain:733",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:8:assign:matmul:734",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:8:assign:quadratic_form:735",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 606,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "606:39:binop:matmul:736",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 606,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "606:39:binop:quadratic_form:737",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 606,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "607:8:assign:assignment:738",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "607:8:assign:kalman_gain:739",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "607:8:assign:matmul:740",
        "kind": "assign",
        "operation": "matmul",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "607:8:assign:prediction_update:741",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "607:8:assign:quadratic_form:742",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "607:26:call:call:743",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1))",
        "line": 607,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "608:12:call:call:744",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 608,
        "column": 12,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "610:20:binop:matmul:745",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 610,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "610:20:binop:quadratic_form:746",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 610,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "611:20:binop:matmul:747",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 611,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "611:20:binop:quadratic_form:748",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 611,
        "column": 20,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "617:8:assign:assignment:749",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 617,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "617:8:assign:innovation_covariance:750",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 617,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "617:8:assign:innovation_update:751",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 617,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "617:27:call:call:752",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 617,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "619:12:subscript:subscript:753",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 619,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "622:8:assign:assignment:754",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 622,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "622:8:assign:innovation_covariance:755",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 622,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "622:8:assign:innovation_update:756",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 622,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "622:22:call:call:757",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 622,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "622:36:call:call:758",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 622,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "623:8:assign:assignment:759",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 623,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "623:8:assign:innovation_update:760",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 623,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "623:24:call:call:761",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 623,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "624:12:call:call:762",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 624,
        "column": 12,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "624:24:call:call:763",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 624,
        "column": 24,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "626:8:assign:assignment:764",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 626,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "626:8:assign:time_step_update:765",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 626,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "627:12:call:call:766",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 627,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "627:38:call:call:767",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 627,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "631:8:return:return:768",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(t + 1, filtered_mean, filtered_factor, log_likelihood + contribution, tf.tensor_scatter_nd_update(increments, tf.reshape(t, (1, 1)), tf.reshape(contribution, (1,))))",
        "line": 631,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "636:12:call:call:769",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.tensor_scatter_nd_update(increments, tf.reshape(t, (1, 1)), tf.reshape(contribution, (1,)))",
        "line": 636,
        "column": 12,
        "evidence": {
          "function": "tf.tensor_scatter_nd_update"
        }
      },
      {
        "id": "638:16:call:call:770",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reshape(t, (1, 1))",
        "line": 638,
        "column": 16,
        "evidence": {
          "function": "tf.reshape"
        }
      },
      {
        "id": "639:16:call:call:771",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reshape(contribution, (1,))",
        "line": 639,
        "column": 16,
        "evidence": {
          "function": "tf.reshape"
        }
      },
      {
        "id": "643:4:assign:assignment:772",
        "kind": "assign",
        "operation": "assignment",
        "target": "_, _, _, log_likelihood, increments",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood",
            "increments"
          ]
        }
      },
      {
        "id": "643:4:assign:innovation_covariance:773",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "_, _, _, log_likelihood, increments",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood",
            "increments"
          ]
        }
      },
      {
        "id": "643:4:assign:kalman_gain:774",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "_, _, _, log_likelihood, increments",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood",
            "increments"
          ]
        }
      },
      {
        "id": "643:4:assign:posterior_or_likelihood:775",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "_, _, _, log_likelihood, increments",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood",
            "increments"
          ]
        }
      },
      {
        "id": "643:4:assign:reparameterization_gradient:776",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "_, _, _, log_likelihood, increments",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood",
            "increments"
          ]
        }
      },
      {
        "id": "643:42:call:call:777",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0, increments0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 643,
        "column": 42,
        "evidence": {
          "function": "tf.while_loop"
        }
      },
      {
        "id": "650:4:return:return:778",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(log_likelihood, increments)",
        "line": 650,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "495:5:subscript:subscript:779",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor]",
        "line": 495,
        "column": 5,
        "evidence": {}
      },
      {
        "id": "674:4:return:return:780",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 674,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "674:11:call:call:781",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 674,
        "column": 11,
        "evidence": {
          "function": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl"
        }
      },
      {
        "id": "674:11:call:posterior_or_likelihood:782",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 674,
        "column": 11,
        "evidence": {
          "function": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl"
        }
      },
      {
        "id": "685:8:call:call:783",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "bool(jitter_updates_filtered_covariance)",
        "line": 685,
        "column": 8,
        "evidence": {
          "function": "bool"
        }
      },
      {
        "id": "653:1:call:call:784",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.function(jit_compile=True)",
        "line": 653,
        "column": 1,
        "evidence": {
          "function": "tf.function"
        }
      },
      {
        "id": "666:5:subscript:subscript:785",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor]",
        "line": 666,
        "column": 5,
        "evidence": {}
      },
      {
        "id": "705:4:return:return:786",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 705,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "705:11:call:call:787",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 705,
        "column": 11,
        "evidence": {
          "function": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl"
        }
      },
      {
        "id": "705:11:call:posterior_or_likelihood:788",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, bool(jitter_updates_filtered_covariance))",
        "line": 705,
        "column": 11,
        "evidence": {
          "function": "_tf_qr_sqrt_factorized_kalman_log_likelihood_impl"
        }
      },
      {
        "id": "716:8:call:call:789",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "bool(jitter_updates_filtered_covariance)",
        "line": 716,
        "column": 8,
        "evidence": {
          "function": "bool"
        }
      },
      {
        "id": "702:5:subscript:subscript:790",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor]",
        "line": 702,
        "column": 5,
        "evidence": {}
      },
      {
        "id": "736:4:assign:assignment:791",
        "kind": "assign",
        "operation": "assignment",
        "target": "value, _increments",
        "expression": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, jitter_updates_filtered_covariance)",
        "line": 736,
        "column": 4,
        "evidence": {
          "targets": [
            "value",
            "_increments"
          ]
        }
      },
      {
        "id": "736:4:assign:innovation_covariance:792",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "value, _increments",
        "expression": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, jitter_updates_filtered_covariance)",
        "line": 736,
        "column": 4,
        "evidence": {
          "targets": [
            "value",
            "_increments"
          ]
        }
      },
      {
        "id": "736:4:assign:posterior_or_likelihood:793",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "value, _increments",
        "expression": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, jitter_updates_filtered_covariance)",
        "line": 736,
        "column": 4,
        "evidence": {
          "targets": [
            "value",
            "_increments"
          ]
        }
      },
      {
        "id": "737:8:call:call:794",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, jitter_updates_filtered_covariance)",
        "line": 737,
        "column": 8,
        "evidence": {
          "function": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments"
        }
      },
      {
        "id": "737:8:call:posterior_or_likelihood:795",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(observations, transition_offset, transition_matrix, transition_noise_factor, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_factor, jitter, jitter_updates_filtered_covariance)",
        "line": 737,
        "column": 8,
        "evidence": {
          "function": "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments"
        }
      },
      {
        "id": "751:4:return:return:796",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "value",
        "line": 751,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "720:1:call:call:797",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.function(jit_compile=True)",
        "line": 720,
        "column": 1,
        "evidence": {
          "function": "tf.function"
        }
      },
      {
        "id": "777:4:assign:assignment:798",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static value inputs')",
        "line": 777,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "777:4:assign:time_step_update:799",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static value inputs')",
        "line": 777,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "777:12:call:call:800",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static value inputs')",
        "line": 777,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "790:4:assign:assignment:801",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 790,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "790:4:assign:time_step_update:802",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 790,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "790:8:call:call:803",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 790,
        "column": 8,
        "evidence": {
          "function": "_as_batched_static_observation_matrix"
        }
      },
      {
        "id": "791:4:assign:assignment:804",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 791,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "791:4:assign:reparameterization_gradient:805",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 791,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "791:18:call:call:806",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 791,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "792:4:assign:assignment:807",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 792,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "792:4:assign:innovation_covariance:808",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 792,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "792:4:assign:time_step_update:809",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 792,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "792:24:call:call:810",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 792,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "793:4:assign:assignment:811",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 793,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "793:4:assign:innovation_covariance:812",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 793,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "793:4:assign:time_step_update:813",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 793,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "793:24:call:call:814",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 793,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "794:4:assign:assignment:815",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 794,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "794:4:assign:innovation_covariance:816",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 794,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "794:4:assign:time_step_update:817",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 794,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "794:28:call:call:818",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 794,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "795:4:assign:assignment:819",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 795,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "795:4:assign:innovation_covariance:820",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 795,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "795:4:assign:time_step_update:821",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 795,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "795:25:call:call:822",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 795,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "796:4:assign:assignment:823",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 796,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "796:4:assign:innovation_covariance:824",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 796,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "796:4:assign:time_step_update:825",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 796,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "796:25:call:call:826",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 796,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "797:4:assign:assignment:827",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 797,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "797:4:assign:innovation_covariance:828",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 797,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "797:4:assign:time_step_update:829",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 797,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "797:29:call:call:830",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 797,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "798:4:assign:assignment:831",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 798,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "798:4:assign:time_step_update:832",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 798,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "798:11:call:call:833",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 798,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "799:4:assign:assignment:834",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 799,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "799:4:assign:innovation_covariance:835",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 799,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "799:4:assign:time_step_update:836",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 799,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "799:31:call:call:837",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 799,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "800:4:call:call:838",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_batched_static_shapes(transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=mean, initial_state_covariance=initial_state_covariance)",
        "line": 800,
        "column": 4,
        "evidence": {
          "function": "_validate_batched_static_shapes"
        }
      },
      {
        "id": "810:4:assign:assignment:839",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 810,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "810:4:assign:innovation_covariance:840",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 810,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "810:4:assign:time_step_update:841",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 810,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "810:20:call:call:842",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 810,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "812:4:assign:assignment:843",
        "kind": "assign",
        "operation": "assignment",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 812,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "812:4:assign:innovation_covariance:844",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 812,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "812:4:assign:shape_reference:845",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 812,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "812:17:subscript:subscript:846",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 812,
        "column": 17,
        "evidence": {}
      },
      {
        "id": "812:17:call:call:847",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 812,
        "column": 17,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "813:4:assign:assignment:848",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 813,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "813:4:assign:innovation_covariance:849",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 813,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "813:4:assign:shape_reference:850",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 813,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "813:16:subscript:subscript:851",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[1]",
        "line": 813,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "813:16:call:call:852",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 813,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "814:4:assign:assignment:853",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 814,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "814:4:assign:innovation_covariance:854",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 814,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "814:4:assign:shape_reference:855",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 814,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "814:14:subscript:subscript:856",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(observation_offset)[1]",
        "line": 814,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "814:14:call:call:857",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observation_offset)",
        "line": 814,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "815:4:assign:assignment:858",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 815,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "815:4:assign:innovation_covariance:859",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 815,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "815:4:assign:time_step_update:860",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 815,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "815:21:call:call:861",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 815,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "816:4:assign:assignment:862",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 816,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "816:4:assign:innovation_covariance:863",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 816,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "816:4:assign:time_step_update:864",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 816,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "816:19:call:call:865",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 816,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "817:4:assign:assignment:866",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 817,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "817:24:call:call:867",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 817,
        "column": 24,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "817:24:call:cholesky:868",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 817,
        "column": 24,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "818:4:assign:assignment:869",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 818,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "818:4:assign:innovation_covariance:870",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 818,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "818:35:call:call:871",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 818,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "818:35:call:cholesky:872",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 818,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "819:4:assign:assignment:873",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 819,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "819:4:assign:innovation_covariance:874",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 819,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "819:36:call:call:875",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 819,
        "column": 36,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "819:36:call:cholesky:876",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 819,
        "column": 36,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "823:4:assign:assignment:877",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else _batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 823,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "823:4:assign:innovation_covariance:878",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else _batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 823,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "826:13:call:call:879",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 826,
        "column": 13,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "826:13:call:cholesky:880",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 826,
        "column": 13,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "828:4:assign:assignment:881",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 828,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "828:4:assign:kalman_gain:882",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 828,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "828:4:assign:posterior_or_likelihood:883",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 828,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "828:4:assign:time_step_update:884",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 828,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "828:21:call:call:885",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 828,
        "column": 21,
        "evidence": {
          "function": "tf.zeros"
        }
      },
      {
        "id": "829:4:assign:assignment:886",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 829,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "829:4:assign:time_step_update:887",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 829,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "829:13:call:call:888",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 829,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "831:4:loop:loop:889",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 831,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "831:13:call:call:890",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 831,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "832:8:assign:assignment:891",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 832,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "832:8:assign:prediction_update:892",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 832,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "832:45:call:call:893",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(transition_matrix, mean)",
        "line": 832,
        "column": 45,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "833:8:assign:assignment:894",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:8:assign:innovation_covariance:895",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:8:assign:kalman_gain:896",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:8:assign:matmul:897",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:8:assign:prediction_update:898",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:8:assign:quadratic_form:899",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "833:27:call:call:900",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 833,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "835:16:binop:matmul:901",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 835,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "835:16:binop:quadratic_form:902",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 835,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "840:8:assign:assignment:903",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 840,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "840:27:call:call:904",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 840,
        "column": 27,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "841:8:assign:assignment:905",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "841:8:assign:matmul:906",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "841:8:assign:prediction_update:907",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "841:8:assign:quadratic_form:908",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "841:31:binop:matmul:909",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "841:31:binop:quadratic_form:910",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "841:50:call:call:911",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(predicted_factor)",
        "line": 841,
        "column": 50,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "843:8:assign:assignment:912",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 843,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "843:8:assign:innovation_update:913",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 843,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "843:8:assign:prediction_update:914",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 843,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "843:21:subscript:subscript:915",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t][tf.newaxis, :]",
        "line": 843,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "843:21:subscript:subscript:916",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 843,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "844:33:call:call:917",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(observation_matrix, predicted_mean)",
        "line": 844,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "846:8:assign:assignment:918",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:innovation_covariance:919",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:innovation_update:920",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:kalman_gain:921",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:matmul:922",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:prediction_update:923",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:8:assign:quadratic_form:924",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "846:27:call:call:925",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 846,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "848:16:binop:matmul:926",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 848,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "848:16:binop:quadratic_form:927",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 848,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "853:8:assign:assignment:928",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 853,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "853:8:assign:innovation_update:929",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 853,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "853:28:call:call:930",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 853,
        "column": 28,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "854:8:assign:assignment:931",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 854,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "854:8:assign:innovation_covariance:932",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 854,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "854:8:assign:innovation_update:933",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 854,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "854:31:call:call:934",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 854,
        "column": 31,
        "evidence": {
          "function": "_batched_factor_solve"
        }
      },
      {
        "id": "855:8:assign:assignment:935",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "855:8:assign:innovation_update:936",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "855:8:assign:kalman_gain:937",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "855:8:assign:matmul:938",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "855:8:assign:prediction_update:939",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "855:8:assign:quadratic_form:940",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 855,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "856:12:binop:matmul:941",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 856,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "856:12:binop:quadratic_form:942",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 856,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "856:12:binop:matmul:943",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix)",
        "line": 856,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "856:12:binop:quadratic_form:944",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix)",
        "line": 856,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "857:14:call:call:945",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(observation_matrix)",
        "line": 857,
        "column": 14,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "861:8:assign:assignment:946",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "861:8:assign:innovation_update:947",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "861:8:assign:kalman_gain:948",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "861:8:assign:prediction_update:949",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "861:8:assign:state_update:950",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "861:41:call:call:951",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 861,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "862:8:assign:assignment:952",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:8:assign:covariance_update:953",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:8:assign:innovation_covariance:954",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:8:assign:kalman_gain:955",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:8:assign:matmul:956",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:8:assign:quadratic_form:957",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 862,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "862:39:binop:matmul:958",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 862,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "862:39:binop:quadratic_form:959",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 862,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "863:8:assign:assignment:960",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:covariance_update:961",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:innovation_covariance:962",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:kalman_gain:963",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:matmul:964",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:prediction_update:965",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:8:assign:quadratic_form:966",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "863:23:call:call:967",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 863,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "865:16:binop:matmul:968",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 865,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "865:16:binop:quadratic_form:969",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 865,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "866:16:binop:matmul:970",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 866,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "866:16:binop:quadratic_form:971",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 866,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "870:8:assign:assignment:972",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 870,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "870:26:call:call:973",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 870,
        "column": 26,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "872:8:assign:assignment:974",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 872,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "872:8:assign:innovation_covariance:975",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 872,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "872:8:assign:innovation_update:976",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 872,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "872:27:call:call:977",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 872,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "874:12:subscript:subscript:978",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[..., tf.newaxis]",
        "line": 874,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "877:8:assign:assignment:979",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 877,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "877:8:assign:innovation_covariance:980",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 877,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "877:8:assign:innovation_update:981",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 877,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "877:22:call:call:982",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 877,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "877:36:call:call:983",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 877,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "878:8:assign:assignment:984",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 878,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "878:8:assign:innovation_update:985",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 878,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "878:24:call:call:986",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 878,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "879:12:call:call:987",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 879,
        "column": 12,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "879:24:call:call:988",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 879,
        "column": 24,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "882:8:assign:assignment:989",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 882,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "882:8:assign:time_step_update:990",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 882,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "883:12:call:call:991",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 883,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "883:38:call:call:992",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 883,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "885:8:assign:assignment:993",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 885,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "886:8:assign:assignment:994",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 886,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "887:8:assign:assignment:995",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 887,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "887:8:assign:kalman_gain:996",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 887,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "887:8:assign:posterior_or_likelihood:997",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 887,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "889:4:return:return:998",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 889,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "913:4:assign:assignment:999",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static while-loop value inputs')",
        "line": 913,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "913:4:assign:time_step_update:1000",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static while-loop value inputs')",
        "line": 913,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "913:12:call:call:1001",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR batched-static while-loop value inputs')",
        "line": 913,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "926:4:assign:assignment:1002",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 926,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "926:4:assign:time_step_update:1003",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 926,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "926:8:call:call:1004",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 926,
        "column": 8,
        "evidence": {
          "function": "_as_batched_static_observation_matrix"
        }
      },
      {
        "id": "927:4:assign:assignment:1005",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 927,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "927:4:assign:innovation_covariance:1006",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 927,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "927:4:assign:reparameterization_gradient:1007",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 927,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "927:4:assign:shape_reference:1008",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "n_timesteps",
        "expression": "tf.shape(y)[0]",
        "line": 927,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "927:18:subscript:subscript:1009",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(y)[0]",
        "line": 927,
        "column": 18,
        "evidence": {}
      },
      {
        "id": "927:18:call:call:1010",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(y)",
        "line": 927,
        "column": 18,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "928:4:assign:assignment:1011",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 928,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "928:4:assign:innovation_covariance:1012",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 928,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "928:4:assign:time_step_update:1013",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 928,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "928:24:call:call:1014",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 928,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "929:4:assign:assignment:1015",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 929,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "929:4:assign:innovation_covariance:1016",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 929,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "929:4:assign:time_step_update:1017",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 929,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "929:24:call:call:1018",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 929,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "930:4:assign:assignment:1019",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 930,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "930:4:assign:innovation_covariance:1020",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 930,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "930:4:assign:time_step_update:1021",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 930,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "930:28:call:call:1022",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 930,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "931:4:assign:assignment:1023",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 931,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "931:4:assign:innovation_covariance:1024",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 931,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "931:4:assign:time_step_update:1025",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 931,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "931:25:call:call:1026",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 931,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "932:4:assign:assignment:1027",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 932,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "932:4:assign:innovation_covariance:1028",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 932,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "932:4:assign:time_step_update:1029",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 932,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "932:25:call:call:1030",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 932,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "933:4:assign:assignment:1031",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 933,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "933:4:assign:innovation_covariance:1032",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 933,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "933:4:assign:time_step_update:1033",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 933,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "933:29:call:call:1034",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 933,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "934:4:assign:assignment:1035",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 934,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "934:4:assign:time_step_update:1036",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean0",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 934,
        "column": 4,
        "evidence": {
          "targets": [
            "mean0"
          ]
        }
      },
      {
        "id": "934:12:call:call:1037",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 934,
        "column": 12,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "935:4:assign:assignment:1038",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 935,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "935:4:assign:innovation_covariance:1039",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 935,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "935:4:assign:time_step_update:1040",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 935,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "935:31:call:call:1041",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 935,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "936:4:call:call:1042",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_batched_static_shapes(transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=mean0, initial_state_covariance=initial_state_covariance)",
        "line": 936,
        "column": 4,
        "evidence": {
          "function": "_validate_batched_static_shapes"
        }
      },
      {
        "id": "946:4:assign:assignment:1043",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 946,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "946:4:assign:innovation_covariance:1044",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 946,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "946:4:assign:time_step_update:1045",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 946,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "946:20:call:call:1046",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 946,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "948:4:assign:assignment:1047",
        "kind": "assign",
        "operation": "assignment",
        "target": "batch_size",
        "expression": "tf.shape(mean0)[0]",
        "line": 948,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "948:4:assign:innovation_covariance:1048",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "batch_size",
        "expression": "tf.shape(mean0)[0]",
        "line": 948,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "948:4:assign:shape_reference:1049",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "batch_size",
        "expression": "tf.shape(mean0)[0]",
        "line": 948,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "948:17:subscript:subscript:1050",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean0)[0]",
        "line": 948,
        "column": 17,
        "evidence": {}
      },
      {
        "id": "948:17:call:call:1051",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean0)",
        "line": 948,
        "column": 17,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "949:4:assign:assignment:1052",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[1]",
        "line": 949,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "949:4:assign:innovation_covariance:1053",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[1]",
        "line": 949,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "949:4:assign:shape_reference:1054",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean0)[1]",
        "line": 949,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "949:16:subscript:subscript:1055",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean0)[1]",
        "line": 949,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "949:16:call:call:1056",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean0)",
        "line": 949,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "950:4:assign:assignment:1057",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 950,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "950:4:assign:innovation_covariance:1058",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 950,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "950:4:assign:shape_reference:1059",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 950,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "950:14:subscript:subscript:1060",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(observation_offset)[1]",
        "line": 950,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "950:14:call:call:1061",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observation_offset)",
        "line": 950,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "951:4:assign:assignment:1062",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 951,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "951:4:assign:innovation_covariance:1063",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 951,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "951:4:assign:time_step_update:1064",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 951,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "951:21:call:call:1065",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 951,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "952:4:assign:assignment:1066",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 952,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "952:4:assign:innovation_covariance:1067",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 952,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "952:4:assign:time_step_update:1068",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 952,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "952:19:call:call:1069",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 952,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "953:4:assign:assignment:1070",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor0",
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 953,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor0"
          ]
        }
      },
      {
        "id": "953:25:call:call:1071",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 953,
        "column": 25,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "953:25:call:cholesky:1072",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 953,
        "column": 25,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "954:4:assign:assignment:1073",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 954,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "954:4:assign:innovation_covariance:1074",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 954,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "954:35:call:call:1075",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 954,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "954:35:call:cholesky:1076",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 954,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "955:4:assign:assignment:1077",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 955,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "955:4:assign:innovation_covariance:1078",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 955,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "955:36:call:call:1079",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 955,
        "column": 36,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "955:36:call:cholesky:1080",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance + jitter_tensor * obs_identity, 0.0)",
        "line": 955,
        "column": 36,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "959:4:assign:assignment:1081",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else _batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 959,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "959:4:assign:innovation_covariance:1082",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else _batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 959,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "962:13:call:call:1083",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 962,
        "column": 13,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "962:13:call:cholesky:1084",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(observation_covariance, 0.0)",
        "line": 962,
        "column": 13,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "964:4:assign:assignment:1085",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood0",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 964,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "964:4:assign:kalman_gain:1086",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood0",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 964,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "964:4:assign:posterior_or_likelihood:1087",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood0",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 964,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "964:4:assign:time_step_update:1088",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood0",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 964,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood0"
          ]
        }
      },
      {
        "id": "964:22:call:call:1089",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 964,
        "column": 22,
        "evidence": {
          "function": "tf.zeros"
        }
      },
      {
        "id": "965:4:assign:assignment:1090",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 965,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "965:4:assign:time_step_update:1091",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 965,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "965:13:call:call:1092",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 965,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "966:4:assign:assignment:1093",
        "kind": "assign",
        "operation": "assignment",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 966,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "966:4:assign:time_step_update:1094",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "t0",
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 966,
        "column": 4,
        "evidence": {
          "targets": [
            "t0"
          ]
        }
      },
      {
        "id": "966:9:call:call:1095",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 966,
        "column": 9,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "969:8:return:return:1096",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "t < n_timesteps",
        "line": 969,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "972:8:assign:assignment:1097",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 972,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "972:8:assign:prediction_update:1098",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 972,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "972:45:call:call:1099",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(transition_matrix, mean)",
        "line": 972,
        "column": 45,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "973:8:assign:assignment:1100",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:8:assign:innovation_covariance:1101",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:8:assign:kalman_gain:1102",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:8:assign:matmul:1103",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:8:assign:prediction_update:1104",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:8:assign:quadratic_form:1105",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "973:27:call:call:1106",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 973,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "975:16:binop:matmul:1107",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 975,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "975:16:binop:quadratic_form:1108",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 975,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "980:8:assign:assignment:1109",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 980,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "980:27:call:call:1110",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 980,
        "column": 27,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "981:8:assign:assignment:1111",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "981:8:assign:matmul:1112",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "981:8:assign:prediction_update:1113",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "981:8:assign:quadratic_form:1114",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "981:31:binop:matmul:1115",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "981:31:binop:quadratic_form:1116",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "981:50:call:call:1117",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(predicted_factor)",
        "line": 981,
        "column": 50,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "983:8:assign:assignment:1118",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 983,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "983:8:assign:innovation_update:1119",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 983,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "983:8:assign:prediction_update:1120",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))",
        "line": 983,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "983:21:subscript:subscript:1121",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t][tf.newaxis, :]",
        "line": 983,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "983:21:subscript:subscript:1122",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 983,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "984:33:call:call:1123",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(observation_matrix, predicted_mean)",
        "line": 984,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "986:8:assign:assignment:1124",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:innovation_covariance:1125",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:innovation_update:1126",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:kalman_gain:1127",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:matmul:1128",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:prediction_update:1129",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:8:assign:quadratic_form:1130",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "986:27:call:call:1131",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 986,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "988:16:binop:matmul:1132",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 988,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "988:16:binop:quadratic_form:1133",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "observation_matrix @ predicted_factor",
        "line": 988,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "993:8:assign:assignment:1134",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 993,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "993:8:assign:innovation_update:1135",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 993,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "993:28:call:call:1136",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 993,
        "column": 28,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "994:8:assign:assignment:1137",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 994,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "994:8:assign:innovation_covariance:1138",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 994,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "994:8:assign:innovation_update:1139",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 994,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "994:31:call:call:1140",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 994,
        "column": 31,
        "evidence": {
          "function": "_batched_factor_solve"
        }
      },
      {
        "id": "995:8:assign:assignment:1141",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "995:8:assign:innovation_update:1142",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "995:8:assign:kalman_gain:1143",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "995:8:assign:matmul:1144",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "995:8:assign:prediction_update:1145",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "995:8:assign:quadratic_form:1146",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 995,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "996:12:binop:matmul:1147",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 996,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "996:12:binop:quadratic_form:1148",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix) @ innovation_precision",
        "line": 996,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "996:12:binop:matmul:1149",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix)",
        "line": 996,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "996:12:binop:quadratic_form:1150",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(observation_matrix)",
        "line": 996,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "997:14:call:call:1151",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(observation_matrix)",
        "line": 997,
        "column": 14,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "1001:8:assign:assignment:1152",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1001:8:assign:innovation_update:1153",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1001:8:assign:kalman_gain:1154",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1001:8:assign:prediction_update:1155",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1001:8:assign:state_update:1156",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1001:41:call:call:1157",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 1001,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1002:8:assign:assignment:1158",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:8:assign:covariance_update:1159",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:8:assign:innovation_covariance:1160",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:8:assign:kalman_gain:1161",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:8:assign:matmul:1162",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:8:assign:quadratic_form:1163",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1002:39:binop:matmul:1164",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1002:39:binop:quadratic_form:1165",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_matrix",
        "line": 1002,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1003:8:assign:assignment:1166",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:covariance_update:1167",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:innovation_covariance:1168",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:kalman_gain:1169",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:matmul:1170",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:prediction_update:1171",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:8:assign:quadratic_form:1172",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1003:23:call:call:1173",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=2)",
        "line": 1003,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1005:16:binop:matmul:1174",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1005,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1005:16:binop:quadratic_form:1175",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1005,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1006:16:binop:matmul:1176",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 1006,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1006:16:binop:quadratic_form:1177",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 1006,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1010:8:assign:assignment:1178",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 1010,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "1010:26:call:call:1179",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 1010,
        "column": 26,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1012:8:assign:assignment:1180",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1012,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1012:8:assign:innovation_covariance:1181",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1012,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1012:8:assign:innovation_update:1182",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1012,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1012:27:call:call:1183",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1012,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "1014:12:subscript:subscript:1184",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[..., tf.newaxis]",
        "line": 1014,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1017:8:assign:assignment:1185",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1017,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1017:8:assign:innovation_covariance:1186",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1017,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1017:8:assign:innovation_update:1187",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1017,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1017:22:call:call:1188",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1017,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1017:36:call:call:1189",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 1017,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "1018:8:assign:assignment:1190",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1018,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1018:8:assign:innovation_update:1191",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1018,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1018:24:call:call:1192",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1018,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1019:12:call:call:1193",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 1019,
        "column": 12,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1019:24:call:call:1194",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 1019,
        "column": 24,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "1022:8:assign:assignment:1195",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 1022,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1022:8:assign:time_step_update:1196",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 1022,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1023:12:call:call:1197",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 1023,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1023:38:call:call:1198",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 1023,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1025:8:return:return:1199",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(t + 1, filtered_mean, filtered_factor, log_likelihood + contribution)",
        "line": 1025,
        "column": 8,
        "evidence": {}
      },
      {
        "id": "1027:4:assign:assignment:1200",
        "kind": "assign",
        "operation": "assignment",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 1027,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1027:4:assign:kalman_gain:1201",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 1027,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1027:4:assign:posterior_or_likelihood:1202",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 1027,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1027:4:assign:reparameterization_gradient:1203",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "_, _, _, log_likelihood",
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 1027,
        "column": 4,
        "evidence": {
          "targets": [
            "_",
            "_",
            "_",
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1027:30:call:call:1204",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.while_loop(cond, body, (t0, mean0, covariance_factor0, log_likelihood0), maximum_iterations=n_timesteps, parallel_iterations=1)",
        "line": 1027,
        "column": 30,
        "evidence": {
          "function": "tf.while_loop"
        }
      },
      {
        "id": "1034:4:return:return:1205",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 1034,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "892:1:call:call:1206",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.function(reduce_retracing=True)",
        "line": 892,
        "column": 1,
        "evidence": {
          "function": "tf.function"
        }
      },
      {
        "id": "1060:4:assign:assignment:1207",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR batched-static value inputs')",
        "line": 1060,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1060:4:assign:time_step_update:1208",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR batched-static value inputs')",
        "line": 1060,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1060:12:call:call:1209",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR batched-static value inputs')",
        "line": 1060,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "1073:4:assign:assignment:1210",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 1073,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1073:4:assign:time_step_update:1211",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 1073,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1073:8:call:call:1212",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_batched_static_observation_matrix(observations, dtype)",
        "line": 1073,
        "column": 8,
        "evidence": {
          "function": "_as_batched_static_observation_matrix"
        }
      },
      {
        "id": "1074:4:assign:assignment:1213",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1074,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1074:4:assign:reparameterization_gradient:1214",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1074,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1074:18:call:call:1215",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 1074,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "1075:4:assign:assignment:1216",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1075,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1075:4:assign:innovation_covariance:1217",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1075,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1075:4:assign:kalman_gain:1218",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1075,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1075:4:assign:time_step_update:1219",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1075,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1075:23:call:call:1220",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1075,
        "column": 23,
        "evidence": {
          "function": "tf.convert_to_tensor"
        }
      },
      {
        "id": "1076:4:call:call:1221",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_mask_shape(y, observation_mask)",
        "line": 1076,
        "column": 4,
        "evidence": {
          "function": "_validate_mask_shape"
        }
      },
      {
        "id": "1077:4:assign:assignment:1222",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1077,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1077:4:assign:innovation_covariance:1223",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1077,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1077:4:assign:time_step_update:1224",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1077,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1077:24:call:call:1225",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1077,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1078:4:assign:assignment:1226",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1078,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1078:4:assign:innovation_covariance:1227",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1078,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1078:4:assign:time_step_update:1228",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1078,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1078:24:call:call:1229",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1078,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1079:4:assign:assignment:1230",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1079,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1079:4:assign:innovation_covariance:1231",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1079,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1079:4:assign:time_step_update:1232",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1079,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1079:28:call:call:1233",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1079,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1080:4:assign:assignment:1234",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1080,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1080:4:assign:innovation_covariance:1235",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1080,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1080:4:assign:time_step_update:1236",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1080,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1080:25:call:call:1237",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1080,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1081:4:assign:assignment:1238",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1081,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1081:4:assign:innovation_covariance:1239",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1081,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1081:4:assign:time_step_update:1240",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1081,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1081:25:call:call:1241",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1081,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1082:4:assign:assignment:1242",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1082,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1082:4:assign:innovation_covariance:1243",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1082,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1082:4:assign:time_step_update:1244",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1082,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1082:29:call:call:1245",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1082,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1083:4:assign:assignment:1246",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1083,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1083:4:assign:time_step_update:1247",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1083,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1083:11:call:call:1248",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1083,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1084:4:assign:assignment:1249",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1084,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1084:4:assign:innovation_covariance:1250",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1084,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1084:4:assign:time_step_update:1251",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1084,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1084:31:call:call:1252",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1084,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1085:4:call:call:1253",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_batched_static_shapes(transition_offset=transition_offset, transition_matrix=transition_matrix, transition_covariance=transition_covariance, observation_offset=observation_offset, observation_matrix=observation_matrix, observation_covariance=observation_covariance, initial_state_mean=mean, initial_state_covariance=initial_state_covariance)",
        "line": 1085,
        "column": 4,
        "evidence": {
          "function": "_validate_batched_static_shapes"
        }
      },
      {
        "id": "1095:4:assign:assignment:1254",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1095,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1095:4:assign:innovation_covariance:1255",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1095,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1095:4:assign:time_step_update:1256",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1095,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1095:20:call:call:1257",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1095,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1097:4:assign:assignment:1258",
        "kind": "assign",
        "operation": "assignment",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 1097,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "1097:4:assign:innovation_covariance:1259",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 1097,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "1097:4:assign:shape_reference:1260",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "batch_size",
        "expression": "tf.shape(mean)[0]",
        "line": 1097,
        "column": 4,
        "evidence": {
          "targets": [
            "batch_size"
          ]
        }
      },
      {
        "id": "1097:17:subscript:subscript:1261",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 1097,
        "column": 17,
        "evidence": {}
      },
      {
        "id": "1097:17:call:call:1262",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 1097,
        "column": 17,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1098:4:assign:assignment:1263",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 1098,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1098:4:assign:innovation_covariance:1264",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 1098,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1098:4:assign:shape_reference:1265",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[1]",
        "line": 1098,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1098:16:subscript:subscript:1266",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[1]",
        "line": 1098,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "1098:16:call:call:1267",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 1098,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1099:4:assign:assignment:1268",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 1099,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1099:4:assign:innovation_covariance:1269",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 1099,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1099:4:assign:shape_reference:1270",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(observation_offset)[1]",
        "line": 1099,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1099:14:subscript:subscript:1271",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(observation_offset)[1]",
        "line": 1099,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "1099:14:call:call:1272",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(observation_offset)",
        "line": 1099,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1100:4:assign:assignment:1273",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1100,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1100:4:assign:innovation_covariance:1274",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1100,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1100:4:assign:time_step_update:1275",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1100,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1100:21:call:call:1276",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1100,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1101:4:assign:assignment:1277",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1101,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1101:4:assign:innovation_covariance:1278",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1101,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1101:4:assign:time_step_update:1279",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1101,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1101:19:call:call:1280",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)",
        "line": 1101,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1102:4:assign:assignment:1281",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1102,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1102:24:call:call:1282",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1102,
        "column": 24,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1102:24:call:cholesky:1283",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1102,
        "column": 24,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1103:4:assign:assignment:1284",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 1103,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1103:4:assign:innovation_covariance:1285",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 1103,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1103:35:call:call:1286",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 1103,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1103:35:call:cholesky:1287",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(transition_covariance, 0.0)",
        "line": 1103,
        "column": 35,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1104:4:assign:assignment:1288",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 1104,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1104:4:assign:kalman_gain:1289",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 1104,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1104:4:assign:posterior_or_likelihood:1290",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 1104,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1104:4:assign:time_step_update:1291",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 1104,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1104:21:call:call:1292",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.zeros((batch_size,), dtype=dtype)",
        "line": 1104,
        "column": 21,
        "evidence": {
          "function": "tf.zeros"
        }
      },
      {
        "id": "1105:4:assign:assignment:1293",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1105,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1105:4:assign:time_step_update:1294",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1105,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1105:13:call:call:1295",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1105,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1106:4:assign:assignment:1296",
        "kind": "assign",
        "operation": "assignment",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1106,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1106:4:assign:time_step_update:1297",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1106,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1106:21:call:call:1298",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1106,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1106:33:call:call:1299",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "math.log(2.0 * math.pi)",
        "line": 1106,
        "column": 33,
        "evidence": {
          "function": "math.log"
        }
      },
      {
        "id": "1108:4:loop:loop:1300",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 1108,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1108:13:call:call:1301",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 1108,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "1109:8:assign:assignment:1302",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 1109,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1109:8:assign:prediction_update:1303",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "transition_offset + _matvec(transition_matrix, mean)",
        "line": 1109,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1109:45:call:call:1304",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(transition_matrix, mean)",
        "line": 1109,
        "column": 45,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1110:8:assign:assignment:1305",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:8:assign:innovation_covariance:1306",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:8:assign:kalman_gain:1307",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:8:assign:matmul:1308",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:8:assign:prediction_update:1309",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:8:assign:quadratic_form:1310",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1110:27:call:call:1311",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((transition_matrix @ covariance_factor, transition_covariance_factor), axis=2)",
        "line": 1110,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1112:16:binop:matmul:1312",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 1112,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1112:16:binop:quadratic_form:1313",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "transition_matrix @ covariance_factor",
        "line": 1112,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1117:8:assign:assignment:1314",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1117,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "1117:27:call:call:1315",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1117,
        "column": 27,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1118:8:assign:assignment:1316",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1118:8:assign:matmul:1317",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1118:8:assign:prediction_update:1318",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1118:8:assign:quadratic_form:1319",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1118:31:binop:matmul:1320",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1118:31:binop:quadratic_form:1321",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ _matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1118:50:call:call:1322",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(predicted_factor)",
        "line": 1118,
        "column": 50,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "1120:8:assign:assignment:1323",
        "kind": "assign",
        "operation": "assignment",
        "target": "base_observation_covariance",
        "expression": "observation_covariance + jitter_tensor * obs_identity",
        "line": 1120,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1120:8:assign:innovation_covariance:1324",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "base_observation_covariance",
        "expression": "observation_covariance + jitter_tensor * obs_identity",
        "line": 1120,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1121:8:assign:assignment:1325",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1121,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1121:8:assign:time_step_update:1326",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1121,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1121:21:call:call:1327",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1121,
        "column": 21,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1121:29:subscript:subscript:1328",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "observation_mask[t]",
        "line": 1121,
        "column": 29,
        "evidence": {}
      },
      {
        "id": "1122:8:assign:assignment:1329",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1122,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1122:8:assign:innovation_covariance:1330",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1122,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1123:8:assign:assignment:1331",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_outer",
        "expression": "row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]",
        "line": 1123,
        "column": 8,
        "evidence": {
          "targets": [
            "row_outer"
          ]
        }
      },
      {
        "id": "1123:20:subscript:subscript:1332",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[:, tf.newaxis]",
        "line": 1123,
        "column": 20,
        "evidence": {}
      },
      {
        "id": "1123:48:subscript:subscript:1333",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[tf.newaxis, :]",
        "line": 1123,
        "column": 48,
        "evidence": {}
      },
      {
        "id": "1124:8:assign:assignment:1334",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_matrix",
        "expression": "observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]",
        "line": 1124,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1124:8:assign:innovation_covariance:1335",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_matrix",
        "expression": "observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]",
        "line": 1124,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1124:8:assign:kalman_gain:1336",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_matrix",
        "expression": "observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]",
        "line": 1124,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1124:57:subscript:subscript:1337",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[tf.newaxis, :, tf.newaxis]",
        "line": 1124,
        "column": 57,
        "evidence": {}
      },
      {
        "id": "1125:8:assign:assignment:1338",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer[tf.newaxis, :, :] + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]",
        "line": 1125,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1125:8:assign:innovation_covariance:1339",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer[tf.newaxis, :, :] + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]",
        "line": 1125,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1125:8:assign:kalman_gain:1340",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer[tf.newaxis, :, :] + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]",
        "line": 1125,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1126:42:subscript:subscript:1341",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_outer[tf.newaxis, :, :]",
        "line": 1126,
        "column": 42,
        "evidence": {}
      },
      {
        "id": "1127:14:subscript:subscript:1342",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.linalg.diag(missing_weight)[tf.newaxis, :, :]",
        "line": 1127,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "1127:14:call:call:1343",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag(missing_weight)",
        "line": 1127,
        "column": 14,
        "evidence": {
          "function": "tf.linalg.diag"
        }
      },
      {
        "id": "1129:8:assign:assignment:1344",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1129,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1129:8:assign:innovation_covariance:1345",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "_batched_cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1129,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1129:40:call:call:1346",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1129,
        "column": 40,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1129:40:call:cholesky:1347",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "_batched_cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1129,
        "column": 40,
        "evidence": {
          "function": "_batched_cholesky_factor"
        }
      },
      {
        "id": "1134:8:assign:assignment:1348",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "(y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))) * row_weight[tf.newaxis, :]",
        "line": 1134,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1134:8:assign:innovation_update:1349",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "(y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))) * row_weight[tf.newaxis, :]",
        "line": 1134,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1134:8:assign:prediction_update:1350",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "(y[t][tf.newaxis, :] - (observation_offset + _matvec(observation_matrix, predicted_mean))) * row_weight[tf.newaxis, :]",
        "line": 1134,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1135:12:subscript:subscript:1351",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t][tf.newaxis, :]",
        "line": 1135,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1135:12:subscript:subscript:1352",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 1135,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1136:36:call:call:1353",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(observation_matrix, predicted_mean)",
        "line": 1136,
        "column": 36,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1137:12:subscript:subscript:1354",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[tf.newaxis, :]",
        "line": 1137,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1138:8:assign:assignment:1355",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:innovation_covariance:1356",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:innovation_update:1357",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:kalman_gain:1358",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:matmul:1359",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:prediction_update:1360",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:8:assign:quadratic_form:1361",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1138:27:call:call:1362",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=2)",
        "line": 1138,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1140:16:binop:matmul:1363",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1140,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1140:16:binop:quadratic_form:1364",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1140,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1145:8:assign:assignment:1365",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1145,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1145:8:assign:innovation_update:1366",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1145,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1145:28:call:call:1367",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1145,
        "column": 28,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1146:8:assign:assignment:1368",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 1146,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1146:8:assign:innovation_covariance:1369",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 1146,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1146:8:assign:innovation_update:1370",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 1146,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1146:31:call:call:1371",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_factor_solve(innovation_factor, obs_identity)",
        "line": 1146,
        "column": 31,
        "evidence": {
          "function": "_batched_factor_solve"
        }
      },
      {
        "id": "1147:8:assign:assignment:1372",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1147:8:assign:innovation_update:1373",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1147:8:assign:kalman_gain:1374",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1147:8:assign:matmul:1375",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1147:8:assign:prediction_update:1376",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1147:8:assign:quadratic_form:1377",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1147,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1148:12:binop:matmul:1378",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1148,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1148:12:binop:quadratic_form:1379",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1148,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1148:12:binop:matmul:1380",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix)",
        "line": 1148,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1148:12:binop:quadratic_form:1381",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ _matrix_transpose(masked_observation_matrix)",
        "line": 1148,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1149:14:call:call:1382",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_transpose(masked_observation_matrix)",
        "line": 1149,
        "column": 14,
        "evidence": {
          "function": "_matrix_transpose"
        }
      },
      {
        "id": "1153:8:assign:assignment:1383",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1153:8:assign:innovation_update:1384",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1153:8:assign:kalman_gain:1385",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1153:8:assign:prediction_update:1386",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1153:8:assign:state_update:1387",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1153:41:call:call:1388",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 1153,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1154:8:assign:assignment:1389",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:8:assign:covariance_update:1390",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:8:assign:innovation_covariance:1391",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:8:assign:kalman_gain:1392",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:8:assign:matmul:1393",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:8:assign:quadratic_form:1394",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1154:39:binop:matmul:1395",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1154:39:binop:quadratic_form:1396",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1154,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1155:8:assign:assignment:1397",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:covariance_update:1398",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:innovation_covariance:1399",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:kalman_gain:1400",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:matmul:1401",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:prediction_update:1402",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:8:assign:quadratic_form:1403",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1155:23:call:call:1404",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=2)",
        "line": 1155,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1157:16:binop:matmul:1405",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1157,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1157:16:binop:quadratic_form:1406",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1157,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1158:16:binop:matmul:1407",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1158,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1158:16:binop:quadratic_form:1408",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1158,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1162:8:assign:assignment:1409",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 1162,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "1162:26:call:call:1410",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_batched_lower_factor_from_horizontal_stack(update_stack)",
        "line": 1162,
        "column": 26,
        "evidence": {
          "function": "_batched_lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1164:8:assign:assignment:1411",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1164,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1164:8:assign:innovation_covariance:1412",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1164,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1164:8:assign:innovation_update:1413",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1164,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1164:27:call:call:1414",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[..., tf.newaxis], lower=True)",
        "line": 1164,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "1166:12:subscript:subscript:1415",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[..., tf.newaxis]",
        "line": 1166,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1169:8:assign:assignment:1416",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1169,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1169:8:assign:innovation_covariance:1417",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1169,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1169:8:assign:innovation_update:1418",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1169,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1169:22:call:call:1419",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])",
        "line": 1169,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1169:36:call:call:1420",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 1169,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "1170:8:assign:assignment:1421",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1170,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1170:8:assign:innovation_update:1422",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1170,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1170:24:call:call:1423",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1)",
        "line": 1170,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1171:12:call:call:1424",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 1171,
        "column": 12,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1171:24:call:call:1425",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 1171,
        "column": 24,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "1174:8:assign:assignment:1426",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1174,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1174:8:assign:innovation_covariance:1427",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1174,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1174:24:call:call:1428",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1174,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1175:8:assign:assignment:1429",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1175,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1175:8:assign:time_step_update:1430",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1175,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1176:12:call:call:1431",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 1176,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1176:38:call:call:1432",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 1176,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1181:8:assign:assignment:1433",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 1181,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1182:8:assign:assignment:1434",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 1182,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1183:8:assign:assignment:1435",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1183,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1183:8:assign:kalman_gain:1436",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1183,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1183:8:assign:posterior_or_likelihood:1437",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1183,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1185:4:return:return:1438",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 1185,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1204:4:assign:assignment:1439",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR compact value inputs')",
        "line": 1204,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1204:4:assign:time_step_update:1440",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR compact value inputs')",
        "line": 1204,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1204:12:call:call:1441",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR compact value inputs')",
        "line": 1204,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "1217:4:assign:assignment:1442",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1217,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1217:4:assign:time_step_update:1443",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1217,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1217:8:call:call:1444",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1217,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "1218:4:assign:assignment:1445",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1218,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1218:4:assign:reparameterization_gradient:1446",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1218,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1218:18:call:call:1447",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 1218,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "1219:4:assign:assignment:1448",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1219,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1219:4:assign:innovation_covariance:1449",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1219,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1219:4:assign:kalman_gain:1450",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1219,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1219:4:assign:time_step_update:1451",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1219,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1219:23:call:call:1452",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1219,
        "column": 23,
        "evidence": {
          "function": "tf.convert_to_tensor"
        }
      },
      {
        "id": "1220:4:call:call:1453",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_mask_shape(y, observation_mask)",
        "line": 1220,
        "column": 4,
        "evidence": {
          "function": "_validate_mask_shape"
        }
      },
      {
        "id": "1221:4:assign:assignment:1454",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1221,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1221:4:assign:innovation_covariance:1455",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1221,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1221:4:assign:time_step_update:1456",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1221,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1221:24:call:call:1457",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1221,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1222:4:assign:assignment:1458",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1222,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1222:4:assign:innovation_covariance:1459",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1222,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1222:4:assign:time_step_update:1460",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1222,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1222:24:call:call:1461",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1222,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1223:4:assign:assignment:1462",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1223,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1223:4:assign:innovation_covariance:1463",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1223,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1223:4:assign:time_step_update:1464",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1223,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1223:28:call:call:1465",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1223,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1224:4:assign:assignment:1466",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1224,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1224:4:assign:innovation_covariance:1467",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1224,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1224:4:assign:time_step_update:1468",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1224,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1224:25:call:call:1469",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1224,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1225:4:assign:assignment:1470",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1225,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1225:4:assign:innovation_covariance:1471",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1225,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1225:4:assign:time_step_update:1472",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1225,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1225:25:call:call:1473",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1225,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1226:4:assign:assignment:1474",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1226,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1226:4:assign:innovation_covariance:1475",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1226,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1226:4:assign:time_step_update:1476",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1226,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1226:29:call:call:1477",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1226,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1227:4:assign:assignment:1478",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1227,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1227:4:assign:time_step_update:1479",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1227,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1227:11:call:call:1480",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1227,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1228:4:assign:assignment:1481",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1228,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1228:4:assign:innovation_covariance:1482",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1228,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1228:4:assign:time_step_update:1483",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1228,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1228:31:call:call:1484",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1228,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1229:4:assign:assignment:1485",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1229,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1229:4:assign:innovation_covariance:1486",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1229,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1229:4:assign:time_step_update:1487",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1229,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1229:20:call:call:1488",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1229,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1231:4:assign:assignment:1489",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1231,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1231:4:assign:innovation_covariance:1490",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1231,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1231:4:assign:shape_reference:1491",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1231,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1231:16:subscript:subscript:1492",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 1231,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "1231:16:call:call:1493",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 1231,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1232:4:assign:assignment:1494",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1232,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1232:4:assign:innovation_covariance:1495",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1232,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1232:4:assign:shape_reference:1496",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1232,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1232:4:assign:time_step_update:1497",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1232,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1232:14:subscript:subscript:1498",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1232,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "1232:14:call:call:1499",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))",
        "line": 1232,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1232:23:call:call:1500",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32))",
        "line": 1232,
        "column": 23,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1232:59:call:call:1501",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 1232,
        "column": 59,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1233:4:assign:assignment:1502",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1233,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1233:4:assign:time_step_update:1503",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1233,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1233:21:call:call:1504",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1233,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1234:4:assign:assignment:1505",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1234,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1234:4:assign:innovation_covariance:1506",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1234,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1234:4:assign:time_step_update:1507",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1234,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1234:19:call:call:1508",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1234,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1235:4:assign:assignment:1509",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1235,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1235:24:call:call:1510",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1235,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1235:24:call:cholesky:1511",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1235,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1236:4:assign:assignment:1512",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1236,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1236:4:assign:kalman_gain:1513",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1236,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1236:4:assign:posterior_or_likelihood:1514",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1236,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1236:4:assign:time_step_update:1515",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1236,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1236:21:call:call:1516",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1236,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1237:4:assign:assignment:1517",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1237,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1237:4:assign:time_step_update:1518",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1237,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1237:13:call:call:1519",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1237,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1238:4:assign:assignment:1520",
        "kind": "assign",
        "operation": "assignment",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1238,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1238:4:assign:time_step_update:1521",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1238,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1238:21:call:call:1522",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1238,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1238:33:call:call:1523",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "math.log(2.0 * math.pi)",
        "line": 1238,
        "column": 33,
        "evidence": {
          "function": "math.log"
        }
      },
      {
        "id": "1240:4:loop:loop:1524",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 1240,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1240:13:call:call:1525",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 1240,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "1241:8:assign:assignment:1526",
        "kind": "assign",
        "operation": "assignment",
        "target": "c",
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1241,
        "column": 8,
        "evidence": {
          "targets": [
            "c"
          ]
        }
      },
      {
        "id": "1241:12:call:call:1527",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1241,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1242:8:assign:assignment:1528",
        "kind": "assign",
        "operation": "assignment",
        "target": "T",
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1242,
        "column": 8,
        "evidence": {
          "targets": [
            "T"
          ]
        }
      },
      {
        "id": "1242:12:call:call:1529",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1242,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1243:8:assign:assignment:1530",
        "kind": "assign",
        "operation": "assignment",
        "target": "Q",
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1243,
        "column": 8,
        "evidence": {
          "targets": [
            "Q"
          ]
        }
      },
      {
        "id": "1243:12:call:call:1531",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1243,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1244:8:assign:assignment:1532",
        "kind": "assign",
        "operation": "assignment",
        "target": "d",
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1244,
        "column": 8,
        "evidence": {
          "targets": [
            "d"
          ]
        }
      },
      {
        "id": "1244:12:call:call:1533",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1244,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1245:8:assign:assignment:1534",
        "kind": "assign",
        "operation": "assignment",
        "target": "Z",
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1245,
        "column": 8,
        "evidence": {
          "targets": [
            "Z"
          ]
        }
      },
      {
        "id": "1245:12:call:call:1535",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1245,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1246:8:assign:assignment:1536",
        "kind": "assign",
        "operation": "assignment",
        "target": "H",
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1246,
        "column": 8,
        "evidence": {
          "targets": [
            "H"
          ]
        }
      },
      {
        "id": "1246:12:call:call:1537",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1246,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1247:8:assign:assignment:1538",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1247,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1247:8:assign:innovation_covariance:1539",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1247,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1247:39:call:call:1540",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1247,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1247:39:call:cholesky:1541",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1247,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1248:8:assign:assignment:1542",
        "kind": "assign",
        "operation": "assignment",
        "target": "base_observation_covariance",
        "expression": "H + jitter_tensor * obs_identity",
        "line": 1248,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1248:8:assign:innovation_covariance:1543",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "base_observation_covariance",
        "expression": "H + jitter_tensor * obs_identity",
        "line": 1248,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1250:8:assign:assignment:1544",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1250,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1250:8:assign:prediction_update:1545",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1250,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1250:29:call:call:1546",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(T, mean)",
        "line": 1250,
        "column": 29,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1251:8:assign:assignment:1547",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:8:assign:innovation_covariance:1548",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:8:assign:kalman_gain:1549",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:8:assign:matmul:1550",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:8:assign:prediction_update:1551",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:8:assign:quadratic_form:1552",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1251:27:call:call:1553",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1251,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1252:13:binop:matmul:1554",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1252,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1252:13:binop:quadratic_form:1555",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1252,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1255:8:assign:assignment:1556",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1255,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "1255:27:call:call:1557",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1255,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1256:8:assign:assignment:1558",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1256:8:assign:matmul:1559",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1256:8:assign:prediction_update:1560",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1256:8:assign:quadratic_form:1561",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1256:31:binop:matmul:1562",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1256:31:binop:quadratic_form:1563",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1256:50:call:call:1564",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 1256,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1258:8:assign:assignment:1565",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1258,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1258:8:assign:time_step_update:1566",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1258,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1258:21:call:call:1567",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1258,
        "column": 21,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1258:29:subscript:subscript:1568",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "observation_mask[t]",
        "line": 1258,
        "column": 29,
        "evidence": {}
      },
      {
        "id": "1259:8:assign:assignment:1569",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1259,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1259:8:assign:innovation_covariance:1570",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1259,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1260:8:assign:assignment:1571",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_outer",
        "expression": "row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]",
        "line": 1260,
        "column": 8,
        "evidence": {
          "targets": [
            "row_outer"
          ]
        }
      },
      {
        "id": "1260:20:subscript:subscript:1572",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[:, tf.newaxis]",
        "line": 1260,
        "column": 20,
        "evidence": {}
      },
      {
        "id": "1260:48:subscript:subscript:1573",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[tf.newaxis, :]",
        "line": 1260,
        "column": 48,
        "evidence": {}
      },
      {
        "id": "1261:8:assign:assignment:1574",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1261,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1261:8:assign:innovation_covariance:1575",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1261,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1261:8:assign:kalman_gain:1576",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1261,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1261:40:subscript:subscript:1577",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[:, tf.newaxis]",
        "line": 1261,
        "column": 40,
        "evidence": {}
      },
      {
        "id": "1262:8:assign:assignment:1578",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1262,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1262:8:assign:innovation_covariance:1579",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1262,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1262:8:assign:kalman_gain:1580",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1262,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1263:54:call:call:1581",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag(missing_weight)",
        "line": 1263,
        "column": 54,
        "evidence": {
          "function": "tf.linalg.diag"
        }
      },
      {
        "id": "1265:8:assign:assignment:1582",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1265,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1265:8:assign:innovation_covariance:1583",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1265,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1265:40:call:call:1584",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1265,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1265:40:call:cholesky:1585",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1265,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1266:8:assign:assignment:1586",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1266,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1266:8:assign:innovation_update:1587",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1266,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1266:8:assign:prediction_update:1588",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1266,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1266:22:subscript:subscript:1589",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 1266,
        "column": 22,
        "evidence": {}
      },
      {
        "id": "1266:34:call:call:1590",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(Z, predicted_mean)",
        "line": 1266,
        "column": 34,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1267:8:assign:assignment:1591",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:innovation_covariance:1592",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:innovation_update:1593",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:kalman_gain:1594",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:matmul:1595",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:prediction_update:1596",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:8:assign:quadratic_form:1597",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1267:27:call:call:1598",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1267,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1268:13:binop:matmul:1599",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1268,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1268:13:binop:quadratic_form:1600",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1268,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1271:8:assign:assignment:1601",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1271,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1271:8:assign:innovation_update:1602",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1271,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1271:28:call:call:1603",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1271,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1272:8:assign:assignment:1604",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1272,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1272:8:assign:innovation_covariance:1605",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1272,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1272:8:assign:innovation_update:1606",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1272,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1272:31:call:call:1607",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1272,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "1273:8:assign:assignment:1608",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1273:8:assign:innovation_update:1609",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1273:8:assign:kalman_gain:1610",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1273:8:assign:matmul:1611",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1273:8:assign:prediction_update:1612",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1273:8:assign:quadratic_form:1613",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1273,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1274:12:binop:matmul:1614",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1274,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1274:12:binop:quadratic_form:1615",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1274,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1274:12:binop:matmul:1616",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix)",
        "line": 1274,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1274:12:binop:quadratic_form:1617",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix)",
        "line": 1274,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1275:14:call:call:1618",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(masked_observation_matrix)",
        "line": 1275,
        "column": 14,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1279:8:assign:assignment:1619",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1279:8:assign:innovation_update:1620",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1279:8:assign:kalman_gain:1621",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1279:8:assign:prediction_update:1622",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1279:8:assign:state_update:1623",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1279:41:call:call:1624",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 1279,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1280:8:assign:assignment:1625",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:8:assign:covariance_update:1626",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:8:assign:innovation_covariance:1627",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:8:assign:kalman_gain:1628",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:8:assign:matmul:1629",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:8:assign:quadratic_form:1630",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1280:39:binop:matmul:1631",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1280:39:binop:quadratic_form:1632",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1280,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1281:8:assign:assignment:1633",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:covariance_update:1634",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:innovation_covariance:1635",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:kalman_gain:1636",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:matmul:1637",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:prediction_update:1638",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:8:assign:quadratic_form:1639",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1281:23:call:call:1640",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1281,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1283:16:binop:matmul:1641",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1283,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1283:16:binop:quadratic_form:1642",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1283,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1284:16:binop:matmul:1643",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1284,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1284:16:binop:quadratic_form:1644",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1284,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1288:8:assign:assignment:1645",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1288,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "1288:26:call:call:1646",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1288,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1290:8:assign:assignment:1647",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1290,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1290:8:assign:innovation_covariance:1648",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1290,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1290:8:assign:innovation_update:1649",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1290,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1290:27:call:call:1650",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1290,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "1292:12:subscript:subscript:1651",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 1292,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1295:8:assign:assignment:1652",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1295,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1295:8:assign:innovation_covariance:1653",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1295,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1295:8:assign:innovation_update:1654",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1295,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1295:22:call:call:1655",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1295,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1295:36:call:call:1656",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 1295,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "1296:8:assign:assignment:1657",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1296,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1296:8:assign:innovation_update:1658",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1296,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1296:24:call:call:1659",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1296,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1296:38:call:call:1660",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 1296,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1296:50:call:call:1661",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 1296,
        "column": 50,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "1297:8:assign:assignment:1662",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1297,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1297:8:assign:innovation_covariance:1663",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1297,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1297:24:call:call:1664",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1297,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1298:8:assign:assignment:1665",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1298,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1298:8:assign:time_step_update:1666",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1298,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1299:12:call:call:1667",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 1299,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1299:38:call:call:1668",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 1299,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1304:8:assign:assignment:1669",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 1304,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1305:8:assign:assignment:1670",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 1305,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1306:8:assign:assignment:1671",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1306,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1306:8:assign:kalman_gain:1672",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1306,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1306:8:assign:posterior_or_likelihood:1673",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1306,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1308:4:return:return:1674",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "log_likelihood",
        "line": 1308,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1327:4:assign:assignment:1675",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR filtered value inputs')",
        "line": 1327,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1327:4:assign:time_step_update:1676",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR filtered value inputs')",
        "line": 1327,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1327:12:call:call:1677",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='QR filtered value inputs')",
        "line": 1327,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "1340:4:assign:assignment:1678",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1340,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1340:4:assign:time_step_update:1679",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1340,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1340:8:call:call:1680",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1340,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "1341:4:assign:assignment:1681",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1341,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1341:4:assign:reparameterization_gradient:1682",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1341,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1341:18:call:call:1683",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 1341,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "1342:4:assign:assignment:1684",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1342,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1342:4:assign:innovation_covariance:1685",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1342,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1342:4:assign:time_step_update:1686",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1342,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1342:24:call:call:1687",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1342,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1343:4:assign:assignment:1688",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1343,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1343:4:assign:innovation_covariance:1689",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1343,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1343:4:assign:time_step_update:1690",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1343,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1343:24:call:call:1691",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1343,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1344:4:assign:assignment:1692",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1344,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1344:4:assign:innovation_covariance:1693",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1344,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1344:4:assign:time_step_update:1694",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1344,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1344:28:call:call:1695",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1344,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1345:4:assign:assignment:1696",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1345,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1345:4:assign:innovation_covariance:1697",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1345,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1345:4:assign:time_step_update:1698",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1345,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1345:25:call:call:1699",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1345,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1346:4:assign:assignment:1700",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1346,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1346:4:assign:innovation_covariance:1701",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1346,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1346:4:assign:time_step_update:1702",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1346,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1346:25:call:call:1703",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1346,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1347:4:assign:assignment:1704",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1347,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1347:4:assign:innovation_covariance:1705",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1347,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1347:4:assign:time_step_update:1706",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1347,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1347:29:call:call:1707",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1347,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1348:4:assign:assignment:1708",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1348,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1348:4:assign:time_step_update:1709",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1348,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1348:11:call:call:1710",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1348,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1349:4:assign:assignment:1711",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1349,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1349:4:assign:innovation_covariance:1712",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1349,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1349:4:assign:time_step_update:1713",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1349,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1349:31:call:call:1714",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1349,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1350:4:assign:assignment:1715",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1350,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1350:4:assign:innovation_covariance:1716",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1350,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1350:4:assign:time_step_update:1717",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1350,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1350:20:call:call:1718",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1350,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1352:4:assign:assignment:1719",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1352,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1352:4:assign:innovation_covariance:1720",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1352,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1352:4:assign:shape_reference:1721",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1352,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1352:16:subscript:subscript:1722",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 1352,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "1352:16:call:call:1723",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 1352,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1353:4:assign:assignment:1724",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1353,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1353:4:assign:innovation_covariance:1725",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1353,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1353:4:assign:shape_reference:1726",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1353,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1353:4:assign:time_step_update:1727",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1353,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1353:14:subscript:subscript:1728",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1353,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "1353:14:call:call:1729",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))",
        "line": 1353,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1353:23:call:call:1730",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32))",
        "line": 1353,
        "column": 23,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1353:59:call:call:1731",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 1353,
        "column": 59,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1354:4:assign:assignment:1732",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1354,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1354:4:assign:time_step_update:1733",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1354,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1354:21:call:call:1734",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1354,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1355:4:assign:assignment:1735",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1355,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1355:4:assign:innovation_covariance:1736",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1355,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1355:4:assign:time_step_update:1737",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1355,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1355:19:call:call:1738",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1355,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1356:4:assign:assignment:1739",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1356,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1356:24:call:call:1740",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1356,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1356:24:call:cholesky:1741",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1356,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1357:4:assign:assignment:1742",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1357,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1357:4:assign:kalman_gain:1743",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1357,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1357:4:assign:posterior_or_likelihood:1744",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1357,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1357:4:assign:time_step_update:1745",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1357,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1357:21:call:call:1746",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1357,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1358:4:assign:assignment:1747",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1358,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1358:4:assign:time_step_update:1748",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1358,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1358:13:call:call:1749",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1358,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1359:4:assign:assignment:1750",
        "kind": "assign",
        "operation": "assignment",
        "target": "means",
        "expression": "[]",
        "line": 1359,
        "column": 4,
        "evidence": {
          "targets": [
            "means"
          ]
        }
      },
      {
        "id": "1360:4:assign:assignment:1751",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariances",
        "expression": "[]",
        "line": 1360,
        "column": 4,
        "evidence": {
          "targets": [
            "covariances"
          ]
        }
      },
      {
        "id": "1362:4:loop:loop:1752",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 1362,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1362:13:call:call:1753",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 1362,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "1363:8:assign:assignment:1754",
        "kind": "assign",
        "operation": "assignment",
        "target": "c",
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1363,
        "column": 8,
        "evidence": {
          "targets": [
            "c"
          ]
        }
      },
      {
        "id": "1363:12:call:call:1755",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1363,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1364:8:assign:assignment:1756",
        "kind": "assign",
        "operation": "assignment",
        "target": "T",
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1364,
        "column": 8,
        "evidence": {
          "targets": [
            "T"
          ]
        }
      },
      {
        "id": "1364:12:call:call:1757",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1364,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1365:8:assign:assignment:1758",
        "kind": "assign",
        "operation": "assignment",
        "target": "Q",
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1365,
        "column": 8,
        "evidence": {
          "targets": [
            "Q"
          ]
        }
      },
      {
        "id": "1365:12:call:call:1759",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1365,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1366:8:assign:assignment:1760",
        "kind": "assign",
        "operation": "assignment",
        "target": "d",
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1366,
        "column": 8,
        "evidence": {
          "targets": [
            "d"
          ]
        }
      },
      {
        "id": "1366:12:call:call:1761",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1366,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1367:8:assign:assignment:1762",
        "kind": "assign",
        "operation": "assignment",
        "target": "Z",
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1367,
        "column": 8,
        "evidence": {
          "targets": [
            "Z"
          ]
        }
      },
      {
        "id": "1367:12:call:call:1763",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1367,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1368:8:assign:assignment:1764",
        "kind": "assign",
        "operation": "assignment",
        "target": "H",
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1368,
        "column": 8,
        "evidence": {
          "targets": [
            "H"
          ]
        }
      },
      {
        "id": "1368:12:call:call:1765",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1368,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1369:8:assign:assignment:1766",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1369,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1369:8:assign:innovation_covariance:1767",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1369,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1369:39:call:call:1768",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1369,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1369:39:call:cholesky:1769",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1369,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1370:8:assign:assignment:1770",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 1370,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1370:8:assign:innovation_covariance:1771",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 1370,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1370:40:call:call:1772",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 1370,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1370:40:call:cholesky:1773",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H + jitter_tensor * obs_identity, 0.0)",
        "line": 1370,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1371:8:assign:assignment:1774",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 1371,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "1371:8:assign:innovation_covariance:1775",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_update_covariance_factor",
        "expression": "observation_covariance_factor if jitter_updates_filtered_covariance else cholesky_factor(H, 0.0)",
        "line": 1371,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_update_covariance_factor"
          ]
        }
      },
      {
        "id": "1374:17:call:call:1776",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 1374,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1374:17:call:cholesky:1777",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(H, 0.0)",
        "line": 1374,
        "column": 17,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1377:8:assign:assignment:1778",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1377,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1377:8:assign:prediction_update:1779",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1377,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1377:29:call:call:1780",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(T, mean)",
        "line": 1377,
        "column": 29,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1378:8:assign:assignment:1781",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:8:assign:innovation_covariance:1782",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:8:assign:kalman_gain:1783",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:8:assign:matmul:1784",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:8:assign:prediction_update:1785",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:8:assign:quadratic_form:1786",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1378:27:call:call:1787",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1378,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1379:13:binop:matmul:1788",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1379,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1379:13:binop:quadratic_form:1789",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1379,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1382:8:assign:assignment:1790",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1382,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "1382:27:call:call:1791",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1382,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1383:8:assign:assignment:1792",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1383:8:assign:matmul:1793",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1383:8:assign:prediction_update:1794",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1383:8:assign:quadratic_form:1795",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1383:31:binop:matmul:1796",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1383:31:binop:quadratic_form:1797",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1383:50:call:call:1798",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 1383,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1385:8:assign:assignment:1799",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 1385,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1385:8:assign:innovation_update:1800",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 1385,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1385:8:assign:prediction_update:1801",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "y[t] - (d + _matvec(Z, predicted_mean))",
        "line": 1385,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1385:21:subscript:subscript:1802",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 1385,
        "column": 21,
        "evidence": {}
      },
      {
        "id": "1385:33:call:call:1803",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(Z, predicted_mean)",
        "line": 1385,
        "column": 33,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1386:8:assign:assignment:1804",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:innovation_covariance:1805",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:innovation_update:1806",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:kalman_gain:1807",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:matmul:1808",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:prediction_update:1809",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:8:assign:quadratic_form:1810",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1386:27:call:call:1811",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((Z @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1386,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1387:13:binop:matmul:1812",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 1387,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1387:13:binop:quadratic_form:1813",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "Z @ predicted_factor",
        "line": 1387,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1390:8:assign:assignment:1814",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1390,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1390:8:assign:innovation_update:1815",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1390,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1390:28:call:call:1816",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1390,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1391:8:assign:assignment:1817",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1391,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1391:8:assign:innovation_covariance:1818",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1391,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1391:8:assign:innovation_update:1819",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1391,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1391:31:call:call:1820",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1391,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "1392:8:assign:assignment:1821",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:8:assign:innovation_update:1822",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:8:assign:kalman_gain:1823",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:8:assign:matmul:1824",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:8:assign:prediction_update:1825",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:8:assign:quadratic_form:1826",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1392:22:binop:matmul:1827",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1392:22:binop:quadratic_form:1828",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z) @ innovation_precision",
        "line": 1392,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1392:22:binop:matmul:1829",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 1392,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1392:22:binop:quadratic_form:1830",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(Z)",
        "line": 1392,
        "column": 22,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1392:45:call:call:1831",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(Z)",
        "line": 1392,
        "column": 45,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1394:8:assign:assignment:1832",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1394:8:assign:innovation_update:1833",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1394:8:assign:kalman_gain:1834",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1394:8:assign:prediction_update:1835",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1394:8:assign:state_update:1836",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1394:41:call:call:1837",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 1394,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1395:8:assign:assignment:1838",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:8:assign:covariance_update:1839",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:8:assign:innovation_covariance:1840",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:8:assign:kalman_gain:1841",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:8:assign:matmul:1842",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:8:assign:quadratic_form:1843",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ Z",
        "line": 1395,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1395:39:binop:matmul:1844",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 1395,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1395:39:binop:quadratic_form:1845",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ Z",
        "line": 1395,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1396:8:assign:assignment:1846",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:covariance_update:1847",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:innovation_covariance:1848",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:kalman_gain:1849",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:matmul:1850",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:prediction_update:1851",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:8:assign:quadratic_form:1852",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1396:23:call:call:1853",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_update_covariance_factor), axis=1)",
        "line": 1396,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1398:16:binop:matmul:1854",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1398,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1398:16:binop:quadratic_form:1855",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1398,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1399:16:binop:matmul:1856",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 1399,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1399:16:binop:quadratic_form:1857",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_update_covariance_factor",
        "line": 1399,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1403:8:assign:assignment:1858",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1403,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "1403:26:call:call:1859",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1403,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1404:8:assign:assignment:1860",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1404:8:assign:matmul:1861",
        "kind": "assign",
        "operation": "matmul",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1404:8:assign:quadratic_form:1862",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1404:30:binop:matmul:1863",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 30,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1404:30:binop:quadratic_form:1864",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 30,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1404:48:call:call:1865",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(filtered_factor)",
        "line": 1404,
        "column": 48,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1406:8:assign:assignment:1866",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1406,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1406:8:assign:innovation_covariance:1867",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1406,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1406:8:assign:innovation_update:1868",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1406,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1406:27:call:call:1869",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1406,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "1408:12:subscript:subscript:1870",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 1408,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1411:8:assign:assignment:1871",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1411,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1411:8:assign:innovation_covariance:1872",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1411,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1411:8:assign:innovation_update:1873",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1411,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1411:22:call:call:1874",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1411,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1411:36:call:call:1875",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 1411,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "1412:8:assign:assignment:1876",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1412,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1412:8:assign:innovation_update:1877",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1412,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1412:24:call:call:1878",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1412,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1412:38:call:call:1879",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 1412,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1412:50:call:call:1880",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 1412,
        "column": 50,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "1413:8:assign:assignment:1881",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 1413,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1413:8:assign:time_step_update:1882",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis)",
        "line": 1413,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1414:12:call:call:1883",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 1414,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1414:38:call:call:1884",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 1414,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1416:8:assign:assignment:1885",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 1416,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1417:8:assign:assignment:1886",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 1417,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1418:8:assign:assignment:1887",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1418,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1418:8:assign:kalman_gain:1888",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1418,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1418:8:assign:posterior_or_likelihood:1889",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1418,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1419:8:call:call:1890",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "means.append(filtered_mean)",
        "line": 1419,
        "column": 8,
        "evidence": {
          "function": "means.append"
        }
      },
      {
        "id": "1420:8:call:call:1891",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "covariances.append(filtered_covariance)",
        "line": 1420,
        "column": 8,
        "evidence": {
          "function": "covariances.append"
        }
      },
      {
        "id": "1422:4:return:return:1892",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(log_likelihood, tf.stack(means, axis=0), tf.stack(covariances, axis=0))",
        "line": 1422,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1422:27:call:call:1893",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.stack(means, axis=0)",
        "line": 1422,
        "column": 27,
        "evidence": {
          "function": "tf.stack"
        }
      },
      {
        "id": "1422:52:call:call:1894",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.stack(covariances, axis=0)",
        "line": 1422,
        "column": 52,
        "evidence": {
          "function": "tf.stack"
        }
      },
      {
        "id": "1324:5:subscript:subscript:1895",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]",
        "line": 1324,
        "column": 5,
        "evidence": {}
      },
      {
        "id": "1442:4:assign:assignment:1896",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR filtered value inputs')",
        "line": 1442,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1442:4:assign:time_step_update:1897",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR filtered value inputs')",
        "line": 1442,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1442:12:call:call:1898",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, transition_offset, transition_matrix, transition_covariance, observation_offset, observation_matrix, observation_covariance, initial_state_mean, initial_state_covariance, jitter, context='masked QR filtered value inputs')",
        "line": 1442,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "1455:4:assign:assignment:1899",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1455,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1455:4:assign:time_step_update:1900",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1455,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1455:8:call:call:1901",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1455,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "1456:4:assign:assignment:1902",
        "kind": "assign",
        "operation": "assignment",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1456,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1456:4:assign:reparameterization_gradient:1903",
        "kind": "assign",
        "operation": "reparameterization_gradient",
        "target": "n_timesteps",
        "expression": "_static_num_timesteps(y)",
        "line": 1456,
        "column": 4,
        "evidence": {
          "targets": [
            "n_timesteps"
          ]
        }
      },
      {
        "id": "1456:18:call:call:1904",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_static_num_timesteps(y)",
        "line": 1456,
        "column": 18,
        "evidence": {
          "function": "_static_num_timesteps"
        }
      },
      {
        "id": "1457:4:assign:assignment:1905",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1457,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1457:4:assign:innovation_covariance:1906",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1457,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1457:4:assign:kalman_gain:1907",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1457,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1457:4:assign:time_step_update:1908",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_mask",
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1457,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_mask"
          ]
        }
      },
      {
        "id": "1457:23:call:call:1909",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.convert_to_tensor(observation_mask, dtype=tf.bool)",
        "line": 1457,
        "column": 23,
        "evidence": {
          "function": "tf.convert_to_tensor"
        }
      },
      {
        "id": "1458:4:call:call:1910",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_validate_mask_shape(y, observation_mask)",
        "line": 1458,
        "column": 4,
        "evidence": {
          "function": "_validate_mask_shape"
        }
      },
      {
        "id": "1459:4:assign:assignment:1911",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1459,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1459:4:assign:innovation_covariance:1912",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1459,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1459:4:assign:time_step_update:1913",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_offset",
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1459,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_offset"
          ]
        }
      },
      {
        "id": "1459:24:call:call:1914",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_offset, dtype)",
        "line": 1459,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1460:4:assign:assignment:1915",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1460,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1460:4:assign:innovation_covariance:1916",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1460,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1460:4:assign:time_step_update:1917",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_matrix",
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1460,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_matrix"
          ]
        }
      },
      {
        "id": "1460:24:call:call:1918",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_matrix, dtype)",
        "line": 1460,
        "column": 24,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1461:4:assign:assignment:1919",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1461,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1461:4:assign:innovation_covariance:1920",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1461,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1461:4:assign:time_step_update:1921",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "transition_covariance",
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1461,
        "column": 4,
        "evidence": {
          "targets": [
            "transition_covariance"
          ]
        }
      },
      {
        "id": "1461:28:call:call:1922",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(transition_covariance, dtype)",
        "line": 1461,
        "column": 28,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1462:4:assign:assignment:1923",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1462,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1462:4:assign:innovation_covariance:1924",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1462,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1462:4:assign:time_step_update:1925",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_offset",
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1462,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_offset"
          ]
        }
      },
      {
        "id": "1462:25:call:call:1926",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_offset, dtype)",
        "line": 1462,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1463:4:assign:assignment:1927",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1463,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1463:4:assign:innovation_covariance:1928",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1463,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1463:4:assign:time_step_update:1929",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_matrix",
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1463,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_matrix"
          ]
        }
      },
      {
        "id": "1463:25:call:call:1930",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_matrix, dtype)",
        "line": 1463,
        "column": 25,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1464:4:assign:assignment:1931",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1464,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1464:4:assign:innovation_covariance:1932",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1464,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1464:4:assign:time_step_update:1933",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "observation_covariance",
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1464,
        "column": 4,
        "evidence": {
          "targets": [
            "observation_covariance"
          ]
        }
      },
      {
        "id": "1464:29:call:call:1934",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(observation_covariance, dtype)",
        "line": 1464,
        "column": 29,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1465:4:assign:assignment:1935",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1465,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1465:4:assign:time_step_update:1936",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "mean",
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1465,
        "column": 4,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1465:11:call:call:1937",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_mean, dtype)",
        "line": 1465,
        "column": 11,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1466:4:assign:assignment:1938",
        "kind": "assign",
        "operation": "assignment",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1466,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1466:4:assign:innovation_covariance:1939",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1466,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1466:4:assign:time_step_update:1940",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "initial_state_covariance",
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1466,
        "column": 4,
        "evidence": {
          "targets": [
            "initial_state_covariance"
          ]
        }
      },
      {
        "id": "1466:31:call:call:1941",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(initial_state_covariance, dtype)",
        "line": 1466,
        "column": 31,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1467:4:assign:assignment:1942",
        "kind": "assign",
        "operation": "assignment",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1467,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1467:4:assign:innovation_covariance:1943",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1467,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1467:4:assign:time_step_update:1944",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "jitter_tensor",
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1467,
        "column": 4,
        "evidence": {
          "targets": [
            "jitter_tensor"
          ]
        }
      },
      {
        "id": "1467:20:call:call:1945",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_to_tensor(jitter, dtype)",
        "line": 1467,
        "column": 20,
        "evidence": {
          "function": "_to_tensor"
        }
      },
      {
        "id": "1469:4:assign:assignment:1946",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1469,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1469:4:assign:innovation_covariance:1947",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1469,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1469:4:assign:shape_reference:1948",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "state_dim",
        "expression": "tf.shape(mean)[0]",
        "line": 1469,
        "column": 4,
        "evidence": {
          "targets": [
            "state_dim"
          ]
        }
      },
      {
        "id": "1469:16:subscript:subscript:1949",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(mean)[0]",
        "line": 1469,
        "column": 16,
        "evidence": {}
      },
      {
        "id": "1469:16:call:call:1950",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(mean)",
        "line": 1469,
        "column": 16,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1470:4:assign:assignment:1951",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1470,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1470:4:assign:innovation_covariance:1952",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1470,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1470:4:assign:shape_reference:1953",
        "kind": "assign",
        "operation": "shape_reference",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1470,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1470:4:assign:time_step_update:1954",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_dim",
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1470,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_dim"
          ]
        }
      },
      {
        "id": "1470:14:subscript:subscript:1955",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]",
        "line": 1470,
        "column": 14,
        "evidence": {}
      },
      {
        "id": "1470:14:call:call:1956",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))",
        "line": 1470,
        "column": 14,
        "evidence": {
          "function": "tf.shape"
        }
      },
      {
        "id": "1470:23:call:call:1957",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32))",
        "line": 1470,
        "column": 23,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1470:59:call:call:1958",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 1470,
        "column": 59,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1471:4:assign:assignment:1959",
        "kind": "assign",
        "operation": "assignment",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1471,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1471:4:assign:time_step_update:1960",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "state_identity",
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1471,
        "column": 4,
        "evidence": {
          "targets": [
            "state_identity"
          ]
        }
      },
      {
        "id": "1471:21:call:call:1961",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(state_dim, dtype=dtype)",
        "line": 1471,
        "column": 21,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1472:4:assign:assignment:1962",
        "kind": "assign",
        "operation": "assignment",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1472,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1472:4:assign:innovation_covariance:1963",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1472,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1472:4:assign:time_step_update:1964",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "obs_identity",
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1472,
        "column": 4,
        "evidence": {
          "targets": [
            "obs_identity"
          ]
        }
      },
      {
        "id": "1472:19:call:call:1965",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.eye(obs_dim, dtype=dtype)",
        "line": 1472,
        "column": 19,
        "evidence": {
          "function": "tf.eye"
        }
      },
      {
        "id": "1473:4:assign:assignment:1966",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1473,
        "column": 4,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1473:24:call:call:1967",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1473,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1473:24:call:cholesky:1968",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(initial_state_covariance, 0.0)",
        "line": 1473,
        "column": 24,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1474:4:assign:assignment:1969",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1474,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1474:4:assign:kalman_gain:1970",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1474,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1474:4:assign:posterior_or_likelihood:1971",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1474,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1474:4:assign:time_step_update:1972",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "log_likelihood",
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1474,
        "column": 4,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1474:21:call:call:1973",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1474,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1475:4:assign:assignment:1974",
        "kind": "assign",
        "operation": "assignment",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1475,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1475:4:assign:time_step_update:1975",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "two_pi",
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1475,
        "column": 4,
        "evidence": {
          "targets": [
            "two_pi"
          ]
        }
      },
      {
        "id": "1475:13:call:call:1976",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(2.0 * math.pi, dtype=dtype)",
        "line": 1475,
        "column": 13,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1476:4:assign:assignment:1977",
        "kind": "assign",
        "operation": "assignment",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1476,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1476:4:assign:time_step_update:1978",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dummy_log_norm",
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1476,
        "column": 4,
        "evidence": {
          "targets": [
            "dummy_log_norm"
          ]
        }
      },
      {
        "id": "1476:21:call:call:1979",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(math.log(2.0 * math.pi), dtype=dtype)",
        "line": 1476,
        "column": 21,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1476:33:call:call:1980",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "math.log(2.0 * math.pi)",
        "line": 1476,
        "column": 33,
        "evidence": {
          "function": "math.log"
        }
      },
      {
        "id": "1477:4:assign:assignment:1981",
        "kind": "assign",
        "operation": "assignment",
        "target": "means",
        "expression": "[]",
        "line": 1477,
        "column": 4,
        "evidence": {
          "targets": [
            "means"
          ]
        }
      },
      {
        "id": "1478:4:assign:assignment:1982",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariances",
        "expression": "[]",
        "line": 1478,
        "column": 4,
        "evidence": {
          "targets": [
            "covariances"
          ]
        }
      },
      {
        "id": "1480:4:loop:loop:1983",
        "kind": "loop",
        "operation": "loop",
        "target": "t",
        "expression": "range(n_timesteps)",
        "line": 1480,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1480:13:call:call:1984",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "range(n_timesteps)",
        "line": 1480,
        "column": 13,
        "evidence": {
          "function": "range"
        }
      },
      {
        "id": "1481:8:assign:assignment:1985",
        "kind": "assign",
        "operation": "assignment",
        "target": "c",
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1481,
        "column": 8,
        "evidence": {
          "targets": [
            "c"
          ]
        }
      },
      {
        "id": "1481:12:call:call:1986",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(transition_offset, t)",
        "line": 1481,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1482:8:assign:assignment:1987",
        "kind": "assign",
        "operation": "assignment",
        "target": "T",
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1482,
        "column": 8,
        "evidence": {
          "targets": [
            "T"
          ]
        }
      },
      {
        "id": "1482:12:call:call:1988",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_matrix, t)",
        "line": 1482,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1483:8:assign:assignment:1989",
        "kind": "assign",
        "operation": "assignment",
        "target": "Q",
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1483,
        "column": 8,
        "evidence": {
          "targets": [
            "Q"
          ]
        }
      },
      {
        "id": "1483:12:call:call:1990",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(transition_covariance, t)",
        "line": 1483,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1484:8:assign:assignment:1991",
        "kind": "assign",
        "operation": "assignment",
        "target": "d",
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1484,
        "column": 8,
        "evidence": {
          "targets": [
            "d"
          ]
        }
      },
      {
        "id": "1484:12:call:call:1992",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_vector_at_time(observation_offset, t)",
        "line": 1484,
        "column": 12,
        "evidence": {
          "function": "_vector_at_time"
        }
      },
      {
        "id": "1485:8:assign:assignment:1993",
        "kind": "assign",
        "operation": "assignment",
        "target": "Z",
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1485,
        "column": 8,
        "evidence": {
          "targets": [
            "Z"
          ]
        }
      },
      {
        "id": "1485:12:call:call:1994",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_matrix, t)",
        "line": 1485,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1486:8:assign:assignment:1995",
        "kind": "assign",
        "operation": "assignment",
        "target": "H",
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1486,
        "column": 8,
        "evidence": {
          "targets": [
            "H"
          ]
        }
      },
      {
        "id": "1486:12:call:call:1996",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matrix_at_time(observation_covariance, t)",
        "line": 1486,
        "column": 12,
        "evidence": {
          "function": "_matrix_at_time"
        }
      },
      {
        "id": "1487:8:assign:assignment:1997",
        "kind": "assign",
        "operation": "assignment",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1487,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1487:8:assign:innovation_covariance:1998",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "transition_covariance_factor",
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1487,
        "column": 8,
        "evidence": {
          "targets": [
            "transition_covariance_factor"
          ]
        }
      },
      {
        "id": "1487:39:call:call:1999",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1487,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1487:39:call:cholesky:2000",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(Q, 0.0)",
        "line": 1487,
        "column": 39,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1488:8:assign:assignment:2001",
        "kind": "assign",
        "operation": "assignment",
        "target": "base_observation_covariance",
        "expression": "H + jitter_tensor * obs_identity",
        "line": 1488,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1488:8:assign:innovation_covariance:2002",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "base_observation_covariance",
        "expression": "H + jitter_tensor * obs_identity",
        "line": 1488,
        "column": 8,
        "evidence": {
          "targets": [
            "base_observation_covariance"
          ]
        }
      },
      {
        "id": "1490:8:assign:assignment:2003",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1490,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1490:8:assign:prediction_update:2004",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_mean",
        "expression": "c + _matvec(T, mean)",
        "line": 1490,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_mean"
          ]
        }
      },
      {
        "id": "1490:29:call:call:2005",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(T, mean)",
        "line": 1490,
        "column": 29,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1491:8:assign:assignment:2006",
        "kind": "assign",
        "operation": "assignment",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:8:assign:innovation_covariance:2007",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:8:assign:kalman_gain:2008",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:8:assign:matmul:2009",
        "kind": "assign",
        "operation": "matmul",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:8:assign:prediction_update:2010",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:8:assign:quadratic_form:2011",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "prediction_stack",
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 8,
        "evidence": {
          "targets": [
            "prediction_stack"
          ]
        }
      },
      {
        "id": "1491:27:call:call:2012",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((T @ covariance_factor, transition_covariance_factor), axis=1)",
        "line": 1491,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1492:13:binop:matmul:2013",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1492,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1492:13:binop:quadratic_form:2014",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "T @ covariance_factor",
        "line": 1492,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1495:8:assign:assignment:2015",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_factor",
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1495,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_factor"
          ]
        }
      },
      {
        "id": "1495:27:call:call:2016",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(prediction_stack)",
        "line": 1495,
        "column": 27,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1496:8:assign:assignment:2017",
        "kind": "assign",
        "operation": "assignment",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1496:8:assign:matmul:2018",
        "kind": "assign",
        "operation": "matmul",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1496:8:assign:prediction_update:2019",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1496:8:assign:quadratic_form:2020",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "predicted_covariance",
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 8,
        "evidence": {
          "targets": [
            "predicted_covariance"
          ]
        }
      },
      {
        "id": "1496:31:binop:matmul:2021",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1496:31:binop:quadratic_form:2022",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_factor @ tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 31,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1496:50:call:call:2023",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(predicted_factor)",
        "line": 1496,
        "column": 50,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1498:8:assign:assignment:2024",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1498,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1498:8:assign:time_step_update:2025",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "row_weight",
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1498,
        "column": 8,
        "evidence": {
          "targets": [
            "row_weight"
          ]
        }
      },
      {
        "id": "1498:21:call:call:2026",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(observation_mask[t], dtype)",
        "line": 1498,
        "column": 21,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1498:29:subscript:subscript:2027",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "observation_mask[t]",
        "line": 1498,
        "column": 29,
        "evidence": {}
      },
      {
        "id": "1499:8:assign:assignment:2028",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1499,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1499:8:assign:innovation_covariance:2029",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_weight",
        "expression": "1.0 - row_weight",
        "line": 1499,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_weight"
          ]
        }
      },
      {
        "id": "1500:8:assign:assignment:2030",
        "kind": "assign",
        "operation": "assignment",
        "target": "row_outer",
        "expression": "row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]",
        "line": 1500,
        "column": 8,
        "evidence": {
          "targets": [
            "row_outer"
          ]
        }
      },
      {
        "id": "1500:20:subscript:subscript:2031",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[:, tf.newaxis]",
        "line": 1500,
        "column": 20,
        "evidence": {}
      },
      {
        "id": "1500:48:subscript:subscript:2032",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[tf.newaxis, :]",
        "line": 1500,
        "column": 48,
        "evidence": {}
      },
      {
        "id": "1501:8:assign:assignment:2033",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1501,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1501:8:assign:innovation_covariance:2034",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1501,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1501:8:assign:kalman_gain:2035",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_matrix",
        "expression": "Z * row_weight[:, tf.newaxis]",
        "line": 1501,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_matrix"
          ]
        }
      },
      {
        "id": "1501:40:subscript:subscript:2036",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "row_weight[:, tf.newaxis]",
        "line": 1501,
        "column": 40,
        "evidence": {}
      },
      {
        "id": "1502:8:assign:assignment:2037",
        "kind": "assign",
        "operation": "assignment",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1502,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1502:8:assign:innovation_covariance:2038",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1502,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1502:8:assign:kalman_gain:2039",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "masked_observation_covariance",
        "expression": "base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)",
        "line": 1502,
        "column": 8,
        "evidence": {
          "targets": [
            "masked_observation_covariance"
          ]
        }
      },
      {
        "id": "1503:54:call:call:2040",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag(missing_weight)",
        "line": 1503,
        "column": 54,
        "evidence": {
          "function": "tf.linalg.diag"
        }
      },
      {
        "id": "1505:8:assign:assignment:2041",
        "kind": "assign",
        "operation": "assignment",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1505,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1505:8:assign:innovation_covariance:2042",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "observation_covariance_factor",
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1505,
        "column": 8,
        "evidence": {
          "targets": [
            "observation_covariance_factor"
          ]
        }
      },
      {
        "id": "1505:40:call:call:2043",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1505,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1505:40:call:cholesky:2044",
        "kind": "call",
        "operation": "cholesky",
        "target": null,
        "expression": "cholesky_factor(masked_observation_covariance, 0.0)",
        "line": 1505,
        "column": 40,
        "evidence": {
          "function": "cholesky_factor"
        }
      },
      {
        "id": "1506:8:assign:assignment:2045",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1506,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1506:8:assign:innovation_update:2046",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1506,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1506:8:assign:prediction_update:2047",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation",
        "expression": "(y[t] - (d + _matvec(Z, predicted_mean))) * row_weight",
        "line": 1506,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation"
          ]
        }
      },
      {
        "id": "1506:22:subscript:subscript:2048",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "y[t]",
        "line": 1506,
        "column": 22,
        "evidence": {}
      },
      {
        "id": "1506:34:call:call:2049",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(Z, predicted_mean)",
        "line": 1506,
        "column": 34,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1507:8:assign:assignment:2050",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:innovation_covariance:2051",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:innovation_update:2052",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:kalman_gain:2053",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:matmul:2054",
        "kind": "assign",
        "operation": "matmul",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:prediction_update:2055",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:8:assign:quadratic_form:2056",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "innovation_stack",
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_stack"
          ]
        }
      },
      {
        "id": "1507:27:call:call:2057",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((masked_observation_matrix @ predicted_factor, observation_covariance_factor), axis=1)",
        "line": 1507,
        "column": 27,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1508:13:binop:matmul:2058",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1508,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1508:13:binop:quadratic_form:2059",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "masked_observation_matrix @ predicted_factor",
        "line": 1508,
        "column": 13,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1511:8:assign:assignment:2060",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1511,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1511:8:assign:innovation_update:2061",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_factor",
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1511,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_factor"
          ]
        }
      },
      {
        "id": "1511:28:call:call:2062",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(innovation_stack)",
        "line": 1511,
        "column": 28,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1512:8:assign:assignment:2063",
        "kind": "assign",
        "operation": "assignment",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1512,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1512:8:assign:innovation_covariance:2064",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1512,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1512:8:assign:innovation_update:2065",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "innovation_precision",
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1512,
        "column": 8,
        "evidence": {
          "targets": [
            "innovation_precision"
          ]
        }
      },
      {
        "id": "1512:31:call:call:2066",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "factor_solve(innovation_factor, obs_identity)",
        "line": 1512,
        "column": 31,
        "evidence": {
          "function": "factor_solve"
        }
      },
      {
        "id": "1513:8:assign:assignment:2067",
        "kind": "assign",
        "operation": "assignment",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1513:8:assign:innovation_update:2068",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1513:8:assign:kalman_gain:2069",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1513:8:assign:matmul:2070",
        "kind": "assign",
        "operation": "matmul",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1513:8:assign:prediction_update:2071",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1513:8:assign:quadratic_form:2072",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "kalman_gain",
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1513,
        "column": 8,
        "evidence": {
          "targets": [
            "kalman_gain"
          ]
        }
      },
      {
        "id": "1514:12:binop:matmul:2073",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1514,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1514:12:binop:quadratic_form:2074",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix) @ innovation_precision",
        "line": 1514,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1514:12:binop:matmul:2075",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix)",
        "line": 1514,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1514:12:binop:quadratic_form:2076",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "predicted_covariance @ tf.transpose(masked_observation_matrix)",
        "line": 1514,
        "column": 12,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1515:14:call:call:2077",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(masked_observation_matrix)",
        "line": 1515,
        "column": 14,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1519:8:assign:assignment:2078",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1519:8:assign:innovation_update:2079",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1519:8:assign:kalman_gain:2080",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1519:8:assign:prediction_update:2081",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1519:8:assign:state_update:2082",
        "kind": "assign",
        "operation": "state_update",
        "target": "filtered_mean",
        "expression": "predicted_mean + _matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_mean"
          ]
        }
      },
      {
        "id": "1519:41:call:call:2083",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_matvec(kalman_gain, innovation)",
        "line": 1519,
        "column": 41,
        "evidence": {
          "function": "_matvec"
        }
      },
      {
        "id": "1520:8:assign:assignment:2084",
        "kind": "assign",
        "operation": "assignment",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:8:assign:covariance_update:2085",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:8:assign:innovation_covariance:2086",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:8:assign:kalman_gain:2087",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:8:assign:matmul:2088",
        "kind": "assign",
        "operation": "matmul",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:8:assign:quadratic_form:2089",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "joseph_left",
        "expression": "state_identity - kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 8,
        "evidence": {
          "targets": [
            "joseph_left"
          ]
        }
      },
      {
        "id": "1520:39:binop:matmul:2090",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1520:39:binop:quadratic_form:2091",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ masked_observation_matrix",
        "line": 1520,
        "column": 39,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1521:8:assign:assignment:2092",
        "kind": "assign",
        "operation": "assignment",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:covariance_update:2093",
        "kind": "assign",
        "operation": "covariance_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:innovation_covariance:2094",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:kalman_gain:2095",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:matmul:2096",
        "kind": "assign",
        "operation": "matmul",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:prediction_update:2097",
        "kind": "assign",
        "operation": "prediction_update",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:8:assign:quadratic_form:2098",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "update_stack",
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 8,
        "evidence": {
          "targets": [
            "update_stack"
          ]
        }
      },
      {
        "id": "1521:23:call:call:2099",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.concat((joseph_left @ predicted_factor, kalman_gain @ observation_covariance_factor), axis=1)",
        "line": 1521,
        "column": 23,
        "evidence": {
          "function": "tf.concat"
        }
      },
      {
        "id": "1523:16:binop:matmul:2100",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1523,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1523:16:binop:quadratic_form:2101",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "joseph_left @ predicted_factor",
        "line": 1523,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1524:16:binop:matmul:2102",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1524,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1524:16:binop:quadratic_form:2103",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "kalman_gain @ observation_covariance_factor",
        "line": 1524,
        "column": 16,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1528:8:assign:assignment:2104",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_factor",
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1528,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_factor"
          ]
        }
      },
      {
        "id": "1528:26:call:call:2105",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "lower_factor_from_horizontal_stack(update_stack)",
        "line": 1528,
        "column": 26,
        "evidence": {
          "function": "lower_factor_from_horizontal_stack"
        }
      },
      {
        "id": "1529:8:assign:assignment:2106",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1529:8:assign:matmul:2107",
        "kind": "assign",
        "operation": "matmul",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1529:8:assign:quadratic_form:2108",
        "kind": "assign",
        "operation": "quadratic_form",
        "target": "filtered_covariance",
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariance"
          ]
        }
      },
      {
        "id": "1529:30:binop:matmul:2109",
        "kind": "binop",
        "operation": "matmul",
        "target": null,
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 30,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1529:30:binop:quadratic_form:2110",
        "kind": "binop",
        "operation": "quadratic_form",
        "target": null,
        "expression": "filtered_factor @ tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 30,
        "evidence": {
          "operator": "MatMult"
        }
      },
      {
        "id": "1529:48:call:call:2111",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.transpose(filtered_factor)",
        "line": 1529,
        "column": 48,
        "evidence": {
          "function": "tf.transpose"
        }
      },
      {
        "id": "1531:8:assign:assignment:2112",
        "kind": "assign",
        "operation": "assignment",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1531,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1531:8:assign:innovation_covariance:2113",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1531,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1531:8:assign:innovation_update:2114",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "solve_innovation",
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1531,
        "column": 8,
        "evidence": {
          "targets": [
            "solve_innovation"
          ]
        }
      },
      {
        "id": "1531:27:call:call:2115",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.triangular_solve(innovation_factor, innovation[:, tf.newaxis], lower=True)",
        "line": 1531,
        "column": 27,
        "evidence": {
          "function": "tf.linalg.triangular_solve"
        }
      },
      {
        "id": "1533:12:subscript:subscript:2116",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "innovation[:, tf.newaxis]",
        "line": 1533,
        "column": 12,
        "evidence": {}
      },
      {
        "id": "1536:8:assign:assignment:2117",
        "kind": "assign",
        "operation": "assignment",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1536,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1536:8:assign:innovation_covariance:2118",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1536,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1536:8:assign:innovation_update:2119",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "mahalanobis",
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1536,
        "column": 8,
        "evidence": {
          "targets": [
            "mahalanobis"
          ]
        }
      },
      {
        "id": "1536:22:call:call:2120",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.square(solve_innovation))",
        "line": 1536,
        "column": 22,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1536:36:call:call:2121",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.square(solve_innovation)",
        "line": 1536,
        "column": 36,
        "evidence": {
          "function": "tf.square"
        }
      },
      {
        "id": "1537:8:assign:assignment:2122",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1537,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1537:8:assign:innovation_update:2123",
        "kind": "assign",
        "operation": "innovation_update",
        "target": "log_det",
        "expression": "2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1537,
        "column": 8,
        "evidence": {
          "targets": [
            "log_det"
          ]
        }
      },
      {
        "id": "1537:24:call:call:2124",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))",
        "line": 1537,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1537:38:call:call:2125",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(tf.linalg.diag_part(innovation_factor))",
        "line": 1537,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1537:50:call:call:2126",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.linalg.diag_part(innovation_factor)",
        "line": 1537,
        "column": 50,
        "evidence": {
          "function": "tf.linalg.diag_part"
        }
      },
      {
        "id": "1538:8:assign:assignment:2127",
        "kind": "assign",
        "operation": "assignment",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1538,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1538:8:assign:innovation_covariance:2128",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "missing_count",
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1538,
        "column": 8,
        "evidence": {
          "targets": [
            "missing_count"
          ]
        }
      },
      {
        "id": "1538:24:call:call:2129",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.reduce_sum(missing_weight)",
        "line": 1538,
        "column": 24,
        "evidence": {
          "function": "tf.reduce_sum"
        }
      },
      {
        "id": "1539:8:assign:assignment:2130",
        "kind": "assign",
        "operation": "assignment",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1539,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1539:8:assign:time_step_update:2131",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "contribution",
        "expression": "-0.5 * (tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis - missing_count * dummy_log_norm)",
        "line": 1539,
        "column": 8,
        "evidence": {
          "targets": [
            "contribution"
          ]
        }
      },
      {
        "id": "1540:12:call:call:2132",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.cast(obs_dim, dtype)",
        "line": 1540,
        "column": 12,
        "evidence": {
          "function": "tf.cast"
        }
      },
      {
        "id": "1540:38:call:call:2133",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.math.log(two_pi)",
        "line": 1540,
        "column": 38,
        "evidence": {
          "function": "tf.math.log"
        }
      },
      {
        "id": "1545:8:assign:assignment:2134",
        "kind": "assign",
        "operation": "assignment",
        "target": "mean",
        "expression": "filtered_mean",
        "line": 1545,
        "column": 8,
        "evidence": {
          "targets": [
            "mean"
          ]
        }
      },
      {
        "id": "1546:8:assign:assignment:2135",
        "kind": "assign",
        "operation": "assignment",
        "target": "covariance_factor",
        "expression": "filtered_factor",
        "line": 1546,
        "column": 8,
        "evidence": {
          "targets": [
            "covariance_factor"
          ]
        }
      },
      {
        "id": "1547:8:assign:assignment:2136",
        "kind": "assign",
        "operation": "assignment",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1547,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1547:8:assign:kalman_gain:2137",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1547,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1547:8:assign:posterior_or_likelihood:2138",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "log_likelihood",
        "expression": "log_likelihood + contribution",
        "line": 1547,
        "column": 8,
        "evidence": {
          "targets": [
            "log_likelihood"
          ]
        }
      },
      {
        "id": "1548:8:call:call:2139",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "means.append(filtered_mean)",
        "line": 1548,
        "column": 8,
        "evidence": {
          "function": "means.append"
        }
      },
      {
        "id": "1549:8:call:call:2140",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "covariances.append(filtered_covariance)",
        "line": 1549,
        "column": 8,
        "evidence": {
          "function": "covariances.append"
        }
      },
      {
        "id": "1551:4:return:return:2141",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "(log_likelihood, tf.stack(means, axis=0), tf.stack(covariances, axis=0))",
        "line": 1551,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1551:27:call:call:2142",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.stack(means, axis=0)",
        "line": 1551,
        "column": 27,
        "evidence": {
          "function": "tf.stack"
        }
      },
      {
        "id": "1551:52:call:call:2143",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.stack(covariances, axis=0)",
        "line": 1551,
        "column": 52,
        "evidence": {
          "function": "tf.stack"
        }
      },
      {
        "id": "1439:5:subscript:subscript:2144",
        "kind": "subscript",
        "operation": "subscript",
        "target": null,
        "expression": "tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]",
        "line": 1439,
        "column": 5,
        "evidence": {}
      },
      {
        "id": "1560:4:return:return:2145",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "FilterRunMetadata(filter_name=filter_name, partition=model.partition, integration_space='full_state', deterministic_completion='none', approximation_label=None, differentiability_status='value_only', compiled_status='tf_function')",
        "line": 1560,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1560:11:call:call:2146",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "FilterRunMetadata(filter_name=filter_name, partition=model.partition, integration_space='full_state', deterministic_completion='none', approximation_label=None, differentiability_status='value_only', compiled_status='tf_function')",
        "line": 1560,
        "column": 11,
        "evidence": {
          "function": "FilterRunMetadata"
        }
      },
      {
        "id": "1578:4:assign:assignment:2147",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "tf.as_dtype(dtype)",
        "line": 1578,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1578:4:assign:time_step_update:2148",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "tf.as_dtype(dtype)",
        "line": 1578,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1578:12:call:call:2149",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.as_dtype(dtype)",
        "line": 1578,
        "column": 12,
        "evidence": {
          "function": "tf.as_dtype"
        }
      },
      {
        "id": "1579:4:return:return:2150",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "TFFilterDiagnostics(backend=backend, mask_convention=mask_convention, regularization=TFRegularizationDiagnostics(jitter=tf.convert_to_tensor(jitter, dtype=dtype), singular_floor=tf.constant(0.0, dtype=dtype), floor_count=tf.constant(0, dtype=tf.int32), psd_projection_residual=tf.constant(0.0, dtype=dtype), implemented_covariance=None, branch_label='qr_square_root', derivative_target='implemented_regularized_law'))",
        "line": 1579,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1579:11:call:call:2151",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "TFFilterDiagnostics(backend=backend, mask_convention=mask_convention, regularization=TFRegularizationDiagnostics(jitter=tf.convert_to_tensor(jitter, dtype=dtype), singular_floor=tf.constant(0.0, dtype=dtype), floor_count=tf.constant(0, dtype=tf.int32), psd_projection_residual=tf.constant(0.0, dtype=dtype), implemented_covariance=None, branch_label='qr_square_root', derivative_target='implemented_regularized_law'))",
        "line": 1579,
        "column": 11,
        "evidence": {
          "function": "TFFilterDiagnostics"
        }
      },
      {
        "id": "1582:23:call:call:2152",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "TFRegularizationDiagnostics(jitter=tf.convert_to_tensor(jitter, dtype=dtype), singular_floor=tf.constant(0.0, dtype=dtype), floor_count=tf.constant(0, dtype=tf.int32), psd_projection_residual=tf.constant(0.0, dtype=dtype), implemented_covariance=None, branch_label='qr_square_root', derivative_target='implemented_regularized_law')",
        "line": 1582,
        "column": 23,
        "evidence": {
          "function": "TFRegularizationDiagnostics"
        }
      },
      {
        "id": "1583:19:call:call:2153",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.convert_to_tensor(jitter, dtype=dtype)",
        "line": 1583,
        "column": 19,
        "evidence": {
          "function": "tf.convert_to_tensor"
        }
      },
      {
        "id": "1584:27:call:call:2154",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1584,
        "column": 27,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1585:24:call:call:2155",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0, dtype=tf.int32)",
        "line": 1585,
        "column": 24,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1586:36:call:call:2156",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf.constant(0.0, dtype=dtype)",
        "line": 1586,
        "column": 36,
        "evidence": {
          "function": "tf.constant"
        }
      },
      {
        "id": "1606:4:assign:assignment:2157",
        "kind": "assign",
        "operation": "assignment",
        "target": "mask",
        "expression": "observation_mask if observation_mask is not None else model.observation_mask",
        "line": 1606,
        "column": 4,
        "evidence": {
          "targets": [
            "mask"
          ]
        }
      },
      {
        "id": "1606:4:assign:innovation_covariance:2158",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mask",
        "expression": "observation_mask if observation_mask is not None else model.observation_mask",
        "line": 1606,
        "column": 4,
        "evidence": {
          "targets": [
            "mask"
          ]
        }
      },
      {
        "id": "1606:4:assign:kalman_gain:2159",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "mask",
        "expression": "observation_mask if observation_mask is not None else model.observation_mask",
        "line": 1606,
        "column": 4,
        "evidence": {
          "targets": [
            "mask"
          ]
        }
      },
      {
        "id": "1607:4:assign:assignment:2160",
        "kind": "assign",
        "operation": "assignment",
        "target": "dtype",
        "expression": "_value_dtype(observations, model.transition_offset, model.transition_matrix, model.transition_covariance, model.observation_offset, model.observation_matrix, model.observation_covariance, model.initial_mean, model.initial_covariance, jitter, context='TensorFlow QR dispatcher inputs')",
        "line": 1607,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1607:4:assign:time_step_update:2161",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "dtype",
        "expression": "_value_dtype(observations, model.transition_offset, model.transition_matrix, model.transition_covariance, model.observation_offset, model.observation_matrix, model.observation_covariance, model.initial_mean, model.initial_covariance, jitter, context='TensorFlow QR dispatcher inputs')",
        "line": 1607,
        "column": 4,
        "evidence": {
          "targets": [
            "dtype"
          ]
        }
      },
      {
        "id": "1607:12:call:call:2162",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_value_dtype(observations, model.transition_offset, model.transition_matrix, model.transition_covariance, model.observation_offset, model.observation_matrix, model.observation_covariance, model.initial_mean, model.initial_covariance, jitter, context='TensorFlow QR dispatcher inputs')",
        "line": 1607,
        "column": 12,
        "evidence": {
          "function": "_value_dtype"
        }
      },
      {
        "id": "1620:4:assign:assignment:2163",
        "kind": "assign",
        "operation": "assignment",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1620,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1620:4:assign:time_step_update:2164",
        "kind": "assign",
        "operation": "time_step_update",
        "target": "y",
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1620,
        "column": 4,
        "evidence": {
          "targets": [
            "y"
          ]
        }
      },
      {
        "id": "1620:8:call:call:2165",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_as_observation_matrix(observations, dtype)",
        "line": 1620,
        "column": 8,
        "evidence": {
          "function": "_as_observation_matrix"
        }
      },
      {
        "id": "1624:16:assign:assignment:2166",
        "kind": "assign",
        "operation": "assignment",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1624,
        "column": 16,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1624:16:assign:innovation_covariance:2167",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1624,
        "column": 16,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1624:62:call:call:2168",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1624,
        "column": 62,
        "evidence": {
          "function": "tf_qr_sqrt_kalman_filter"
        }
      },
      {
        "id": "1635:55:call:call:2169",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "bool(jitter_updates_filtered_covariance)",
        "line": 1635,
        "column": 55,
        "evidence": {
          "function": "bool"
        }
      },
      {
        "id": "1639:16:assign:assignment:2170",
        "kind": "assign",
        "operation": "assignment",
        "target": "value",
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1639,
        "column": 16,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1639:16:assign:posterior_or_likelihood:2171",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "value",
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1639,
        "column": 16,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1639:24:call:call:2172",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1639,
        "column": 24,
        "evidence": {
          "function": "tf_qr_sqrt_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1639:24:call:posterior_or_likelihood:2173",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, jitter=jitter, jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance))",
        "line": 1639,
        "column": 24,
        "evidence": {
          "function": "tf_qr_sqrt_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1650:55:call:call:2174",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "bool(jitter_updates_filtered_covariance)",
        "line": 1650,
        "column": 55,
        "evidence": {
          "function": "bool"
        }
      },
      {
        "id": "1653:16:assign:assignment:2175",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_means",
        "expression": "None",
        "line": 1653,
        "column": 16,
        "evidence": {
          "targets": [
            "filtered_means"
          ]
        }
      },
      {
        "id": "1654:16:assign:assignment:2176",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariances",
        "expression": "None",
        "line": 1654,
        "column": 16,
        "evidence": {
          "targets": [
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1655:12:assign:assignment:2177",
        "kind": "assign",
        "operation": "assignment",
        "target": "filter_name",
        "expression": "'tf_qr_sqrt_kalman'",
        "line": 1655,
        "column": 12,
        "evidence": {
          "targets": [
            "filter_name"
          ]
        }
      },
      {
        "id": "1656:12:assign:assignment:2178",
        "kind": "assign",
        "operation": "assignment",
        "target": "mask_convention",
        "expression": "'none'",
        "line": 1656,
        "column": 12,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1656:12:assign:kalman_gain:2179",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "mask_convention",
        "expression": "'none'",
        "line": 1656,
        "column": 12,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1659:16:assign:assignment:2180",
        "kind": "assign",
        "operation": "assignment",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1659,
        "column": 16,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1659:16:assign:innovation_covariance:2181",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1659,
        "column": 16,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1659:62:call:call:2182",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1659,
        "column": 62,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_filter"
        }
      },
      {
        "id": "1673:16:assign:assignment:2183",
        "kind": "assign",
        "operation": "assignment",
        "target": "value",
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1673,
        "column": 16,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1673:16:assign:posterior_or_likelihood:2184",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "value",
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1673,
        "column": 16,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1673:24:call:call:2185",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1673,
        "column": 24,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1673:24:call:posterior_or_likelihood:2186",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1673,
        "column": 24,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1686:16:assign:assignment:2187",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_means",
        "expression": "None",
        "line": 1686,
        "column": 16,
        "evidence": {
          "targets": [
            "filtered_means"
          ]
        }
      },
      {
        "id": "1687:16:assign:assignment:2188",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariances",
        "expression": "None",
        "line": 1687,
        "column": 16,
        "evidence": {
          "targets": [
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1688:12:assign:assignment:2189",
        "kind": "assign",
        "operation": "assignment",
        "target": "filter_name",
        "expression": "'tf_qr_sqrt_masked_kalman'",
        "line": 1688,
        "column": 12,
        "evidence": {
          "targets": [
            "filter_name"
          ]
        }
      },
      {
        "id": "1689:12:assign:assignment:2190",
        "kind": "assign",
        "operation": "assignment",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1689,
        "column": 12,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1689:12:assign:innovation_covariance:2191",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1689,
        "column": 12,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1689:12:assign:kalman_gain:2192",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1689,
        "column": 12,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1692:18:call:call:2193",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError('tf_masked_qr requires an observation mask')",
        "line": 1692,
        "column": 18,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "1694:12:assign:assignment:2194",
        "kind": "assign",
        "operation": "assignment",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1694,
        "column": 12,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1694:12:assign:innovation_covariance:2195",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "value, filtered_means, filtered_covariances",
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1694,
        "column": 12,
        "evidence": {
          "targets": [
            "value",
            "filtered_means",
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1694:58:call:call:2196",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_filter(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1694,
        "column": 58,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_filter"
        }
      },
      {
        "id": "1708:12:assign:assignment:2197",
        "kind": "assign",
        "operation": "assignment",
        "target": "value",
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1708,
        "column": 12,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1708:12:assign:posterior_or_likelihood:2198",
        "kind": "assign",
        "operation": "posterior_or_likelihood",
        "target": "value",
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1708,
        "column": 12,
        "evidence": {
          "targets": [
            "value"
          ]
        }
      },
      {
        "id": "1708:20:call:call:2199",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1708,
        "column": 20,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1708:20:call:posterior_or_likelihood:2200",
        "kind": "call",
        "operation": "posterior_or_likelihood",
        "target": null,
        "expression": "tf_qr_sqrt_masked_kalman_log_likelihood_compact(observations=y, transition_offset=model.transition_offset, transition_matrix=model.transition_matrix, transition_covariance=model.transition_covariance, observation_offset=model.observation_offset, observation_matrix=model.observation_matrix, observation_covariance=model.observation_covariance, initial_state_mean=model.initial_mean, initial_state_covariance=model.initial_covariance, observation_mask=mask, jitter=jitter)",
        "line": 1708,
        "column": 20,
        "evidence": {
          "function": "tf_qr_sqrt_masked_kalman_log_likelihood_compact"
        }
      },
      {
        "id": "1721:12:assign:assignment:2201",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_means",
        "expression": "None",
        "line": 1721,
        "column": 12,
        "evidence": {
          "targets": [
            "filtered_means"
          ]
        }
      },
      {
        "id": "1722:12:assign:assignment:2202",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariances",
        "expression": "None",
        "line": 1722,
        "column": 12,
        "evidence": {
          "targets": [
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1723:8:assign:assignment:2203",
        "kind": "assign",
        "operation": "assignment",
        "target": "filter_name",
        "expression": "'tf_qr_sqrt_masked_kalman'",
        "line": 1723,
        "column": 8,
        "evidence": {
          "targets": [
            "filter_name"
          ]
        }
      },
      {
        "id": "1724:8:assign:assignment:2204",
        "kind": "assign",
        "operation": "assignment",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1724,
        "column": 8,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1724:8:assign:innovation_covariance:2205",
        "kind": "assign",
        "operation": "innovation_covariance",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1724,
        "column": 8,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1724:8:assign:kalman_gain:2206",
        "kind": "assign",
        "operation": "kalman_gain",
        "target": "mask_convention",
        "expression": "'static_dummy_row'",
        "line": 1724,
        "column": 8,
        "evidence": {
          "targets": [
            "mask_convention"
          ]
        }
      },
      {
        "id": "1726:14:call:call:2207",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "ValueError(f'unknown TensorFlow QR linear Gaussian backend: {backend}')",
        "line": 1726,
        "column": 14,
        "evidence": {
          "function": "ValueError"
        }
      },
      {
        "id": "1729:8:assign:assignment:2208",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_means",
        "expression": "None",
        "line": 1729,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_means"
          ]
        }
      },
      {
        "id": "1730:8:assign:assignment:2209",
        "kind": "assign",
        "operation": "assignment",
        "target": "filtered_covariances",
        "expression": "None",
        "line": 1730,
        "column": 8,
        "evidence": {
          "targets": [
            "filtered_covariances"
          ]
        }
      },
      {
        "id": "1732:4:return:return:2210",
        "kind": "return",
        "operation": "return",
        "target": null,
        "expression": "TFFilterValueResult(log_likelihood=value, filtered_means=filtered_means, filtered_covariances=filtered_covariances, metadata=_metadata(filter_name=filter_name, model=model), diagnostics=_diagnostics(backend=backend, mask_convention=mask_convention, jitter=jitter, dtype=dtype))",
        "line": 1732,
        "column": 4,
        "evidence": {}
      },
      {
        "id": "1732:11:call:call:2211",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "TFFilterValueResult(log_likelihood=value, filtered_means=filtered_means, filtered_covariances=filtered_covariances, metadata=_metadata(filter_name=filter_name, model=model), diagnostics=_diagnostics(backend=backend, mask_convention=mask_convention, jitter=jitter, dtype=dtype))",
        "line": 1732,
        "column": 11,
        "evidence": {
          "function": "TFFilterValueResult"
        }
      },
      {
        "id": "1736:17:call:call:2212",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_metadata(filter_name=filter_name, model=model)",
        "line": 1736,
        "column": 17,
        "evidence": {
          "function": "_metadata"
        }
      },
      {
        "id": "1737:20:call:call:2213",
        "kind": "call",
        "operation": "call",
        "target": null,
        "expression": "_diagnostics(backend=backend, mask_convention=mask_convention, jitter=jitter, dtype=dtype)",
        "line": 1737,
        "column": 20,
        "evidence": {
          "function": "_diagnostics"
        }
      }
    ],
    "diagnostics": [],
    "metadata": {
      "schema_version": "1.0",
      "contract": "ast_operation_graph"
    }
  },
  "recommended_actions": [
    {
      "kind": "fix_or_explain_missing_recursion_operation",
      "target": "qr",
      "severity": "high"
    },
    {
      "kind": "fix_or_explain_missing_recursion_operation",
      "target": "triangular_solve",
      "severity": "high"
    },
    {
      "kind": "add_or_explain_missing_shape_guard",
      "target": "covariance_guard",
      "severity": "medium"
    }
  ],
  "metadata": {
    "schema_version": "1.0",
    "contract": "kalman_recursion_audit"
  }
}
=== C2: audit-math-to-code downdate identity vs srukf_factor_tf.py ===
{
  "status": "structural_mismatch",
  "workflow": "audit_math_to_code",
  "question": "Does this code implement the supplied math?",
  "claim_class": "math_to_code",
  "answer": "Code is missing required equation terms or has structural conflicts.",
  "evidence": [
    {
      "id": "audit_math_to_code:structural_mismatch",
      "class": "structural_mismatch",
      "source": "structural_matcher",
      "summary": "Code is missing required equation terms or has structural conflicts.",
      "low_level": {
        "status": "mismatch",
        "reason": "Code is missing required equation terms or has structural conflicts.",
        "equation": "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T",
        "matched_terms": [],
        "missing_terms": [
          "K",
          "S_f",
          "S_pred",
          "S_y",
          "T"
        ],
        "extra_code_terms": [
          "Callable",
          "Mapping",
          "MappingProxyType",
          "TFSRUKFMapFn",
          "TFSRUKFParameterDerivativeFn",
          "TFSRUKFPointJacobianFn",
          "TFSRUKFSigmaPointRule",
          "TFSRUKFStepDerivatives",
          "TFSRUKFStepResult",
          "Tensor",
          "ValueError",
          "__setattr__",
          "_as_matrix",
          "_as_vector",
          "_factor_derivative_residual",
          "_factor_from_centered",
          "_factor_reconstruction_residual",
          "_logdet_from_lower_factor",
          "_right_solve_spd",
          "_weighted_covariance",
          "_weighted_covariance_first_derivative",
          "_weighted_mean",
          "abs",
          "append",
          "assert_greater_equal",
          "augmented_factor",
          "augmented_mean",
          "axis_weight",
          "branch_label",
          "cast",
          "centered",
          "centered_left",
          "centered_observation",
          "centered_right",
          "centered_state",
          "cholesky_factor_first_derivatives",
          "concat",
          "constant",
          "control_dependencies",
          "convert_to_tensor",
          "covariance",
          "covariance_factor",
          "covariance_weights",
          "cross_covariance",
          "d_augmented_factor",
          "d_augmented_mean",
          "d_centered_left",
          "d_centered_observation",
          "d_centered_right",
          "d_centered_state",
          "d_covariance",
          "d_cross_covariance",
          "d_factor",
          "d_filtered_covariance",
          "d_filtered_covariance_rows",
          "d_filtered_factor",
          "d_filtered_mean",
          "d_gain",
          "d_gain_rows",
          "d_innovation",
          "d_innovation_covariance",
          "d_observation_direct",
          "d_observation_fn",
          "d_observation_points",
          "d_points",
          "d_predicted_mean",
          "d_predicted_observation",
          "d_s",
          "d_state_covariance",
          "d_state_points",
          "d_transition_direct",
          "d_transition_fn",
          "dataclass",
          "debugging",
          "derivatives",
          "diag_part",
          "diagnostics",
          "dict",
          "dim",
          "einsum",
          "eye",
          "factor",
          "factor_solve",
          "fill",
          "filtered_covariance",
          "filtered_factor",
          "filtered_jitter",
          "filtered_mean",
          "float",
          "float64",
          "gain",
          "getattr",
          "identity",
          "innovation",
          "innovation_covariance",
          "innovation_factor",
          "int",
          "int32",
          "linalg",
          "log",
          "log_likelihood",
          "lower_factor_from_horizontal_stack",
          "math",
          "matrix",
          "matvec",
          "mean_weights",
          "name",
          "newaxis",
          "norm",
          "object",
          "observation",
          "observation_fn",
          "observation_jacobian",
          "observation_jacobian_fn",
          "observation_points",
          "offsets",
          "parameter_dim",
          "parameter_index",
          "pi",
          "point_count",
          "points",
          "predicted_mean",
          "predicted_observation",
          "property",
          "quadratic_term",
          "range",
          "rank",
          "reconstructed",
          "reduce_max",
          "reduce_min",
          "reduce_sum",
          "residuals",
          "rhs",
          "rule",
          "scale",
          "scaled",
          "score",
          "score_terms",
          "self",
          "shape",
          "solve_weight",
          "solved_t",
          "sqrt",
          "stack",
          "state_covariance",
          "state_factor",
          "state_points",
          "str",
          "symmetrize",
          "tensor",
          "tensordot",
          "tf",
          "tf_srukf_unit_sigma_point_rule",
          "trace",
          "trace_term",
          "transition_fn",
          "transition_jacobian",
          "transition_jacobian_fn",
          "transpose",
          "value",
          "weights",
          "zeros"
        ],
        "trace_map": {
          "equation_terms": [
            "K",
            "S_f",
            "S_pred",
            "S_y",
            "T"
          ],
          "alias_map": {},
          "mapped_terms": [
            "K",
            "S_f",
            "S_pred",
            "S_y",
            "T"
          ],
          "term_traces": [
            {
              "equation_term": "K",
              "mapped_code_term": "K",
              "matched": false,
              "source": "direct"
            },
            {
              "equation_term": "S_f",
              "mapped_code_term": "S_f",
              "matched": false,
              "source": "direct"
            },
            {
              "equation_term": "S_pred",
              "mapped_code_term": "S_pred",
              "matched": false,
              "source": "direct"
            },
            {
              "equation_term": "S_y",
              "mapped_code_term": "S_y",
              "matched": false,
              "source": "direct"
            },
            {
              "equation_term": "T",
              "mapped_code_term": "T",
              "matched": false,
              "source": "direct"
            }
          ],
          "alias_collisions": [],
          "matched_terms": [],
          "missing_terms": [
            "K",
            "S_f",
            "S_pred",
            "S_y",
            "T"
          ],
          "extra_code_terms": [
            "Callable",
            "Mapping",
            "MappingProxyType",
            "TFSRUKFMapFn",
            "TFSRUKFParameterDerivativeFn",
            "TFSRUKFPointJacobianFn",
            "TFSRUKFSigmaPointRule",
            "TFSRUKFStepDerivatives",
            "TFSRUKFStepResult",
            "Tensor",
            "ValueError",
            "__setattr__",
            "_as_matrix",
            "_as_vector",
            "_factor_derivative_residual",
            "_factor_from_centered",
            "_factor_reconstruction_residual",
            "_logdet_from_lower_factor",
            "_right_solve_spd",
            "_weighted_covariance",
            "_weighted_covariance_first_derivative",
            "_weighted_mean",
            "abs",
            "append",
            "assert_greater_equal",
            "augmented_factor",
            "augmented_mean",
            "axis_weight",
            "branch_label",
            "cast",
            "centered",
            "centered_left",
            "centered_observation",
            "centered_right",
            "centered_state",
            "cholesky_factor_first_derivatives",
            "concat",
            "constant",
            "control_dependencies",
            "convert_to_tensor",
            "covariance",
            "covariance_factor",
            "covariance_weights",
            "cross_covariance",
            "d_augmented_factor",
            "d_augmented_mean",
            "d_centered_left",
            "d_centered_observation",
            "d_centered_right",
            "d_centered_state",
            "d_covariance",
            "d_cross_covariance",
            "d_factor",
            "d_filtered_covariance",
            "d_filtered_covariance_rows",
            "d_filtered_factor",
            "d_filtered_mean",
            "d_gain",
            "d_gain_rows",
            "d_innovation",
            "d_innovation_covariance",
            "d_observation_direct",
            "d_observation_fn",
            "d_observation_points",
            "d_points",
            "d_predicted_mean",
            "d_predicted_observation",
            "d_s",
            "d_state_covariance",
            "d_state_points",
            "d_transition_direct",
            "d_transition_fn",
            "dataclass",
            "debugging",
            "derivatives",
            "diag_part",
            "diagnostics",
            "dict",
            "dim",
            "einsum",
            "eye",
            "factor",
            "factor_solve",
            "fill",
            "filtered_covariance",
            "filtered_factor",
            "filtered_jitter",
            "filtered_mean",
            "float",
            "float64",
            "gain",
            "getattr",
            "identity",
            "innovation",
            "innovation_covariance",
            "innovation_factor",
            "int",
            "int32",
            "linalg",
            "log",
            "log_likelihood",
            "lower_factor_from_horizontal_stack",
            "math",
            "matrix",
            "matvec",
            "mean_weights",
            "name",
            "newaxis",
            "norm",
            "object",
            "observation",
            "observation_fn",
            "observation_jacobian",
            "observation_jacobian_fn",
            "observation_points",
            "offsets",
            "parameter_dim",
            "parameter_index",
            "pi",
            "point_count",
            "points",
            "predicted_mean",
            "predicted_observation",
            "property",
            "quadratic_term",
            "range",
            "rank",
            "reconstructed",
            "reduce_max",
            "reduce_min",
            "reduce_sum",
            "residuals",
            "rhs",
            "rule",
            "scale",
            "scaled",
            "score",
            "score_terms",
            "self",
            "shape",
            "solve_weight",
            "solved_t",
            "sqrt",
            "stack",
            "state_covariance",
            "state_factor",
            "state_points",
            "str",
            "symmetrize",
            "tensor",
            "tensordot",
            "tf",
            "tf_srukf_unit_sigma_point_rule",
            "trace",
            "trace_term",
            "transition_fn",
            "transition_jacobian",
            "transition_jacobian_fn",
            "transpose",
            "value",
            "weights",
            "zeros"
          ],
          "code_operations": {
            "names": [
              "Callable",
              "Mapping",
              "MappingProxyType",
              "TFSRUKFMapFn",
              "TFSRUKFParameterDerivativeFn",
              "TFSRUKFPointJacobianFn",
              "TFSRUKFSigmaPointRule",
              "TFSRUKFStepDerivatives",
              "TFSRUKFStepResult",
              "Tensor",
              "ValueError",
              "__setattr__",
              "_as_matrix",
              "_as_vector",
              "_factor_derivative_residual",
              "_factor_from_centered",
              "_factor_reconstruction_residual",
              "_logdet_from_lower_factor",
              "_right_solve_spd",
              "_weighted_covariance",
              "_weighted_covariance_first_derivative",
              "_weighted_mean",
              "abs",
              "append",
              "assert_greater_equal",
              "augmented_factor",
              "augmented_mean",
              "axis_weight",
              "branch_label",
              "cast",
              "centered",
              "centered_left",
              "centered_observation",
              "centered_right",
              "centered_state",
              "cholesky_factor_first_derivatives",
              "concat",
              "constant",
              "control_dependencies",
              "convert_to_tensor",
              "covariance",
              "covariance_factor",
              "covariance_weights",
              "cross_covariance",
              "d_augmented_factor",
              "d_augmented_mean",
              "d_centered_left",
              "d_centered_observation",
              "d_centered_right",
              "d_centered_state",
              "d_covariance",
              "d_cross_covariance",
              "d_factor",
              "d_filtered_covariance",
              "d_filtered_covariance_rows",
              "d_filtered_factor",
              "d_filtered_mean",
              "d_gain",
              "d_gain_rows",
              "d_innovation",
              "d_innovation_covariance",
              "d_observation_direct",
              "d_observation_fn",
              "d_observation_points",
              "d_points",
              "d_predicted_mean",
              "d_predicted_observation",
              "d_s",
              "d_state_covariance",
              "d_state_points",
              "d_transition_direct",
              "d_transition_fn",
              "dataclass",
              "debugging",
              "derivatives",
              "diag_part",
              "diagnostics",
              "dict",
              "dim",
              "einsum",
              "eye",
              "factor",
              "factor_solve",
              "fill",
              "filtered_covariance",
              "filtered_factor",
              "filtered_jitter",
              "filtered_mean",
              "float",
              "float64",
              "gain",
              "getattr",
              "identity",
              "innovation",
              "innovation_covariance",
              "innovation_factor",
              "int",
              "int32",
              "linalg",
              "log",
              "log_likelihood",
              "lower_factor_from_horizontal_stack",
              "math",
              "matrix",
              "matvec",
              "mean_weights",
              "name",
              "newaxis",
              "norm",
              "object",
              "observation",
              "observation_fn",
              "observation_jacobian",
              "observation_jacobian_fn",
              "observation_points",
              "offsets",
              "parameter_dim",
              "parameter_index",
              "pi",
              "point_count",
              "points",
              "predicted_mean",
              "predicted_observation",
              "property",
              "quadratic_term",
              "range",
              "rank",
              "reconstructed",
              "reduce_max",
              "reduce_min",
              "reduce_sum",
              "residuals",
              "rhs",
              "rule",
              "scale",
              "scaled",
              "score",
              "score_terms",
              "self",
              "shape",
              "solve_weight",
              "solved_t",
              "sqrt",
              "stack",
              "state_covariance",
              "state_factor",
              "state_points",
              "str",
              "symmetrize",
              "tensor",
              "tensordot",
              "tf",
              "tf_srukf_unit_sigma_point_rule",
              "trace",
              "trace_term",
              "transition_fn",
              "transition_jacobian",
              "transition_jacobian_fn",
              "transpose",
              "value",
              "weights",
              "zeros"
            ],
            "calls": [
              "MappingProxyType",
              "TFSRUKFSigmaPointRule",
              "TFSRUKFStepResult",
              "ValueError",
              "__setattr__",
              "_as_matrix",
              "_as_vector",
              "_factor_derivative_residual",
              "_factor_from_centered",
              "_factor_reconstruction_residual",
              "_logdet_from_lower_factor",
              "_right_solve_spd",
              "_weighted_covariance",
              "_weighted_covariance_first_derivative",
              "_weighted_mean",
              "abs",
              "append",
              "assert_greater_equal",
              "cast",
              "cholesky_factor_first_derivatives",
              "concat",
              "constant",
              "control_dependencies",
              "convert_to_tensor",
              "d_observation_fn",
              "d_transition_fn",
              "dataclass",
              "diag_part",
              "dict",
              "einsum",
              "eye",
              "factor_solve",
              "fill",
              "getattr",
              "identity",
              "int",
              "log",
              "lower_factor_from_horizontal_stack",
              "matvec",
              "norm",
              "observation_fn",
              "observation_jacobian_fn",
              "range",
              "reduce_max",
              "reduce_min",
              "reduce_sum",
              "shape",
              "sqrt",
              "stack",
              "symmetrize",
              "tensordot",
              "tf_srukf_unit_sigma_point_rule",
              "trace",
              "transition_fn",
              "transition_jacobian_fn",
              "transpose",
              "zeros"
            ],
            "operators": [
              "Add",
              "BitOr",
              "Div",
              "MatMult",
              "Mult",
              "Sub"
            ],
            "function_args": {
              "tf_srukf_unit_sigma_point_rule": [
                "dim"
              ],
              "_as_vector": [
                "value"
              ],
              "_as_matrix": [
                "value"
              ],
              "_weighted_mean": [
                "points",
                "weights"
              ],
              "_weighted_covariance": [
                "centered_left",
                "centered_right",
                "weights"
              ],
              "_weighted_covariance_first_derivative": [
                "centered_left",
                "centered_right",
                "d_centered_left",
                "d_centered_right",
                "weights"
              ],
              "_factor_from_centered": [
                "centered",
                "weights"
              ],
              "_factor_reconstruction_residual": [
                "factor",
                "covariance"
              ],
              "_factor_derivative_residual": [
                "factor",
                "d_factor",
                "d_covariance"
              ],
              "_right_solve_spd": [
                "covariance_factor",
                "matrix"
              ],
              "_logdet_from_lower_factor": [
                "factor"
              ],
              "tf_srukf_factor_score_step": [
                "observation",
                "augmented_mean",
                "augmented_factor"
              ],
              "__post_init__": [
                "self"
              ],
              "point_count": [
                "self"
              ],
              "parameter_dim": [
                "self"
              ]
            }
          },
          "boundary": "This trace map records structural term visibility only. It is not semantic proof that the code implements the documented math.",
          "scope_diagnostic": {
            "status": "not_applicable",
            "math_scope_terms": [],
            "mapped_math_scope_terms": [],
            "code_function_args": [
              "augmented_factor",
              "augmented_mean",
              "centered",
              "centered_left",
              "centered_right",
              "covariance",
              "covariance_factor",
              "d_centered_left",
              "d_centered_right",
              "d_covariance",
              "d_factor",
              "dim",
              "factor",
              "matrix",
              "observation",
              "points",
              "self",
              "value",
              "weights"
            ],
            "matched_scope_terms": [],
            "missing_scope_terms": [],
            "function_level_markers": [],
            "supports": "No mathematical function-argument or conditioning boundary was detected for static scope comparison.",
            "does_not_support": "This classification says nothing about semantic code correctness.",
            "safe_wording": "Treat scope comparison as not applicable; retain the ordinary term and conflict diagnostics.",
            "boundary": "Scope diagnostics are structural and non-executing; they are not code correctness proof."
          }
        },
        "conflicts": [],
        "code_summary": {
          "names": [
            "Callable",
            "Mapping",
            "MappingProxyType",
            "TFSRUKFMapFn",
            "TFSRUKFParameterDerivativeFn",
            "TFSRUKFPointJacobianFn",
            "TFSRUKFSigmaPointRule",
            "TFSRUKFStepDerivatives",
            "TFSRUKFStepResult",
            "Tensor",
            "ValueError",
            "__setattr__",
            "_as_matrix",
            "_as_vector",
            "_factor_derivative_residual",
            "_factor_from_centered",
            "_factor_reconstruction_residual",
            "_logdet_from_lower_factor",
            "_right_solve_spd",
            "_weighted_covariance",
            "_weighted_covariance_first_derivative",
            "_weighted_mean",
            "abs",
            "append",
            "assert_greater_equal",
            "augmented_factor",
            "augmented_mean",
            "axis_weight",
            "branch_label",
            "cast",
            "centered",
            "centered_left",
            "centered_observation",
            "centered_right",
            "centered_state",
            "cholesky_factor_first_derivatives",
            "concat",
            "constant",
            "control_dependencies",
            "convert_to_tensor",
            "covariance",
            "covariance_factor",
            "covariance_weights",
            "cross_covariance",
            "d_augmented_factor",
            "d_augmented_mean",
            "d_centered_left",
            "d_centered_observation",
            "d_centered_right",
            "d_centered_state",
            "d_covariance",
            "d_cross_covariance",
            "d_factor",
            "d_filtered_covariance",
            "d_filtered_covariance_rows",
            "d_filtered_factor",
            "d_filtered_mean",
            "d_gain",
            "d_gain_rows",
            "d_innovation",
            "d_innovation_covariance",
            "d_observation_direct",
            "d_observation_fn",
            "d_observation_points",
            "d_points",
            "d_predicted_mean",
            "d_predicted_observation",
            "d_s",
            "d_state_covariance",
            "d_state_points",
            "d_transition_direct",
            "d_transition_fn",
            "dataclass",
            "debugging",
            "derivatives",
            "diag_part",
            "diagnostics",
            "dict",
            "dim",
            "einsum",
            "eye",
            "factor",
            "factor_solve",
            "fill",
            "filtered_covariance",
            "filtered_factor",
            "filtered_jitter",
            "filtered_mean",
            "float",
            "float64",
            "gain",
            "getattr",
            "identity",
            "innovation",
            "innovation_covariance",
            "innovation_factor",
            "int",
            "int32",
            "linalg",
            "log",
            "log_likelihood",
            "lower_factor_from_horizontal_stack",
            "math",
            "matrix",
            "matvec",
            "mean_weights",
            "name",
            "newaxis",
            "norm",
            "object",
            "observation",
            "observation_fn",
            "observation_jacobian",
            "observation_jacobian_fn",
            "observation_points",
            "offsets",
            "parameter_dim",
            "parameter_index",
            "pi",
            "point_count",
            "points",
            "predicted_mean",
            "predicted_observation",
            "property",
            "quadratic_term",
            "range",
            "rank",
            "reconstructed",
            "reduce_max",
            "reduce_min",
            "reduce_sum",
            "residuals",
            "rhs",
            "rule",
            "scale",
            "scaled",
            "score",
            "score_terms",
            "self",
            "shape",
            "solve_weight",
            "solved_t",
            "sqrt",
            "stack",
            "state_covariance",
            "state_factor",
            "state_points",
            "str",
            "symmetrize",
            "tensor",
            "tensordot",
            "tf",
            "tf_srukf_unit_sigma_point_rule",
            "trace",
            "trace_term",
            "transition_fn",
            "transition_jacobian",
            "transition_jacobian_fn",
            "transpose",
            "value",
            "weights",
            "zeros"
          ],
          "calls": [
            "MappingProxyType",
            "TFSRUKFSigmaPointRule",
            "TFSRUKFStepResult",
            "ValueError",
            "__setattr__",
            "_as_matrix",
            "_as_vector",
            "_factor_derivative_residual",
            "_factor_from_centered",
            "_factor_reconstruction_residual",
            "_logdet_from_lower_factor",
            "_right_solve_spd",
            "_weighted_covariance",
            "_weighted_covariance_first_derivative",
            "_weighted_mean",
            "abs",
            "append",
            "assert_greater_equal",
            "cast",
            "cholesky_factor_first_derivatives",
            "concat",
            "constant",
            "control_dependencies",
            "convert_to_tensor",
            "d_observation_fn",
            "d_transition_fn",
            "dataclass",
            "diag_part",
            "dict",
            "einsum",
            "eye",
            "factor_solve",
            "fill",
            "getattr",
            "identity",
            "int",
            "log",
            "lower_factor_from_horizontal_stack",
            "matvec",
            "norm",
            "observation_fn",
            "observation_jacobian_fn",
            "range",
            "reduce_max",
            "reduce_min",
            "reduce_sum",
            "shape",
            "sqrt",
            "stack",
            "symmetrize",
            "tensordot",
            "tf_srukf_unit_sigma_point_rule",
            "trace",
            "transition_fn",
            "transition_jacobian_fn",
            "transpose",
            "zeros"
          ],
          "operators": [
            "Add",
            "BitOr",
            "Div",
            "MatMult",
            "Mult",
            "Sub"
          ],
          "function_args": {
            "tf_srukf_unit_sigma_point_rule": [
              "dim"
            ],
            "_as_vector": [
              "value"
            ],
            "_as_matrix": [
              "value"
            ],
            "_weighted_mean": [
              "points",
              "weights"
            ],
            "_weighted_covariance": [
              "centered_left",
              "centered_right",
              "weights"
            ],
            "_weighted_covariance_first_derivative": [
              "centered_left",
              "centered_right",
              "d_centered_left",
              "d_centered_right",
              "weights"
            ],
            "_factor_from_centered": [
              "centered",
              "weights"
            ],
            "_factor_reconstruction_residual": [
              "factor",
              "covariance"
            ],
            "_factor_derivative_residual": [
              "factor",
              "d_factor",
              "d_covariance"
            ],
            "_right_solve_spd": [
              "covariance_factor",
              "matrix"
            ],
            "_logdet_from_lower_factor": [
              "factor"
            ],
            "tf_srukf_factor_score_step": [
              "observation",
              "augmented_mean",
              "augmented_factor"
            ],
            "__post_init__": [
              "self"
            ],
            "point_count": [
              "self"
            ],
            "parameter_dim": [
              "self"
            ]
          }
        },
        "workbench_result": {
          "question": {
            "question_type": "code_implements_equation",
            "target": "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T",
            "givens": [],
            "assumptions": [],
            "context": {
              "aliases": {}
            },
            "metadata": {
              "schema_version": "1.0",
              "contract": "math_debugging_question"
            }
          },
          "status": "refuted",
          "reason": "Code is missing required equation terms or has structural conflicts.",
          "obligations": [],
          "assumptions": [],
          "backend_attempts": [],
          "counterexamples": [],
          "actions": [
            {
              "kind": "inspect_missing_or_conflicting_code_terms"
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
          "contract": "equation_code_match_result"
        }
      }
    }
  ],
  "evidence_classes": [
    "structural_mismatch"
  ],
  "certification_source": "none",
  "veto_reasons": [],
  "assumptions": [],
  "counterexamples": [],
  "actions": [
    {
      "code": "human_review",
      "description": "Inspect structural mismatch or scope limitation."
    },
    {
      "code": "human_review",
      "description": "Review structural matches, missing terms, aliases, and audit-only extras before treating code as correct."
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
      "code": "structural_evidence_not_proof",
      "text": "Structural evidence is not a semantic proof."
    }
  ],
  "evidence_ledger": {
    "version": "1.0",
    "scope": "scoped_high_level_workflow_result",
    "provenance": {
      "workflow": "audit_math_to_code",
      "status": "structural_mismatch",
      "certification_source": "none",
      "evidence_classes": [
        "structural_mismatch"
      ]
    },
    "evidence_items": [
      {
        "id": "audit_math_to_code:structural_mismatch",
        "class": "structural_mismatch",
        "source": "structural_matcher",
        "summary": "Code is missing required equation terms or has structural conflicts.",
        "low_level": {
          "status": "mismatch",
          "reason": "Code is missing required equation terms or has structural conflicts.",
          "equation": "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T",
          "matched_terms": [],
          "missing_terms": [
            "K",
            "S_f",
            "S_pred",
            "S_y",
            "T"
          ],
          "extra_code_terms": [
            "Callable",
            "Mapping",
            "MappingProxyType",
            "TFSRUKFMapFn",
            "TFSRUKFParameterDerivativeFn",
            "TFSRUKFPointJacobianFn",
            "TFSRUKFSigmaPointRule",
            "TFSRUKFStepDerivatives",
            "TFSRUKFStepResult",
            "Tensor",
            "ValueError",
            "__setattr__",
            "_as_matrix",
            "_as_vector",
            "_factor_derivative_residual",
            "_factor_from_centered",
            "_factor_reconstruction_residual",
            "_logdet_from_lower_factor",
            "_right_solve_spd",
            "_weighted_covariance",
            "_weighted_covariance_first_derivative",
            "_weighted_mean",
            "abs",
            "append",
            "assert_greater_equal",
            "augmented_factor",
            "augmented_mean",
            "axis_weight",
            "branch_label",
            "cast",
            "centered",
            "centered_left",
            "centered_observation",
            "centered_right",
            "centered_state",
            "cholesky_factor_first_derivatives",
            "concat",
            "constant",
            "control_dependencies",
            "convert_to_tensor",
            "covariance",
            "covariance_factor",
            "covariance_weights",
            "cross_covariance",
            "d_augmented_factor",
            "d_augmented_mean",
            "d_centered_left",
            "d_centered_observation",
            "d_centered_right",
            "d_centered_state",
            "d_covariance",
            "d_cross_covariance",
            "d_factor",
            "d_filtered_covariance",
            "d_filtered_covariance_rows",
            "d_filtered_factor",
            "d_filtered_mean",
            "d_gain",
            "d_gain_rows",
            "d_innovation",
            "d_innovation_covariance",
            "d_observation_direct",
            "d_observation_fn",
            "d_observation_points",
            "d_points",
            "d_predicted_mean",
            "d_predicted_observation",
            "d_s",
            "d_state_covariance",
            "d_state_points",
            "d_transition_direct",
            "d_transition_fn",
            "dataclass",
            "debugging",
            "derivatives",
            "diag_part",
            "diagnostics",
            "dict",
            "dim",
            "einsum",
            "eye",
            "factor",
            "factor_solve",
            "fill",
            "filtered_covariance",
            "filtered_factor",
            "filtered_jitter",
            "filtered_mean",
            "float",
            "float64",
            "gain",
            "getattr",
            "identity",
            "innovation",
            "innovation_covariance",
            "innovation_factor",
            "int",
            "int32",
            "linalg",
            "log",
            "log_likelihood",
            "lower_factor_from_horizontal_stack",
            "math",
            "matrix",
            "matvec",
            "mean_weights",
            "name",
            "newaxis",
            "norm",
            "object",
            "observation",
            "observation_fn",
            "observation_jacobian",
            "observation_jacobian_fn",
            "observation_points",
            "offsets",
            "parameter_dim",
            "parameter_index",
            "pi",
            "point_count",
            "points",
            "predicted_mean",
            "predicted_observation",
            "property",
            "quadratic_term",
            "range",
            "rank",
            "reconstructed",
            "reduce_max",
            "reduce_min",
            "reduce_sum",
            "residuals",
            "rhs",
            "rule",
            "scale",
            "scaled",
            "score",
            "score_terms",
            "self",
            "shape",
            "solve_weight",
            "solved_t",
            "sqrt",
            "stack",
            "state_covariance",
            "state_factor",
            "state_points",
            "str",
            "symmetrize",
            "tensor",
            "tensordot",
            "tf",
            "tf_srukf_unit_sigma_point_rule",
            "trace",
            "trace_term",
            "transition_fn",
            "transition_jacobian",
            "transition_jacobian_fn",
            "transpose",
            "value",
            "weights",
            "zeros"
          ],
          "trace_map": {
            "equation_terms": [
              "K",
              "S_f",
              "S_pred",
              "S_y",
              "T"
            ],
            "alias_map": {},
            "mapped_terms": [
              "K",
              "S_f",
              "S_pred",
              "S_y",
              "T"
            ],
            "term_traces": [
              {
                "equation_term": "K",
                "mapped_code_term": "K",
                "matched": false,
                "source": "direct"
              },
              {
                "equation_term": "S_f",
                "mapped_code_term": "S_f",
                "matched": false,
                "source": "direct"
              },
              {
                "equation_term": "S_pred",
                "mapped_code_term": "S_pred",
                "matched": false,
                "source": "direct"
              },
              {
                "equation_term": "S_y",
                "mapped_code_term": "S_y",
                "matched": false,
                "source": "direct"
              },
              {
                "equation_term": "T",
                "mapped_code_term": "T",
                "matched": false,
                "source": "direct"
              }
            ],
            "alias_collisions": [],
            "matched_terms": [],
            "missing_terms": [
              "K",
              "S_f",
              "S_pred",
              "S_y",
              "T"
            ],
            "extra_code_terms": [
              "Callable",
              "Mapping",
              "MappingProxyType",
              "TFSRUKFMapFn",
              "TFSRUKFParameterDerivativeFn",
              "TFSRUKFPointJacobianFn",
              "TFSRUKFSigmaPointRule",
              "TFSRUKFStepDerivatives",
              "TFSRUKFStepResult",
              "Tensor",
              "ValueError",
              "__setattr__",
              "_as_matrix",
              "_as_vector",
              "_factor_derivative_residual",
              "_factor_from_centered",
              "_factor_reconstruction_residual",
              "_logdet_from_lower_factor",
              "_right_solve_spd",
              "_weighted_covariance",
              "_weighted_covariance_first_derivative",
              "_weighted_mean",
              "abs",
              "append",
              "assert_greater_equal",
              "augmented_factor",
              "augmented_mean",
              "axis_weight",
              "branch_label",
              "cast",
              "centered",
              "centered_left",
              "centered_observation",
              "centered_right",
              "centered_state",
              "cholesky_factor_first_derivatives",
              "concat",
              "constant",
              "control_dependencies",
              "convert_to_tensor",
              "covariance",
              "covariance_factor",
              "covariance_weights",
              "cross_covariance",
              "d_augmented_factor",
              "d_augmented_mean",
              "d_centered_left",
              "d_centered_observation",
              "d_centered_right",
              "d_centered_state",
              "d_covariance",
              "d_cross_covariance",
              "d_factor",
              "d_filtered_covariance",
              "d_filtered_covariance_rows",
              "d_filtered_factor",
              "d_filtered_mean",
              "d_gain",
              "d_gain_rows",
              "d_innovation",
              "d_innovation_covariance",
              "d_observation_direct",
              "d_observation_fn",
              "d_observation_points",
              "d_points",
              "d_predicted_mean",
              "d_predicted_observation",
              "d_s",
              "d_state_covariance",
              "d_state_points",
              "d_transition_direct",
              "d_transition_fn",
              "dataclass",
              "debugging",
              "derivatives",
              "diag_part",
              "diagnostics",
              "dict",
              "dim",
              "einsum",
              "eye",
              "factor",
              "factor_solve",
              "fill",
              "filtered_covariance",
              "filtered_factor",
              "filtered_jitter",
              "filtered_mean",
              "float",
              "float64",
              "gain",
              "getattr",
              "identity",
              "innovation",
              "innovation_covariance",
              "innovation_factor",
              "int",
              "int32",
              "linalg",
              "log",
              "log_likelihood",
              "lower_factor_from_horizontal_stack",
              "math",
              "matrix",
              "matvec",
              "mean_weights",
              "name",
              "newaxis",
              "norm",
              "object",
              "observation",
              "observation_fn",
              "observation_jacobian",
              "observation_jacobian_fn",
              "observation_points",
              "offsets",
              "parameter_dim",
              "parameter_index",
              "pi",
              "point_count",
              "points",
              "predicted_mean",
              "predicted_observation",
              "property",
              "quadratic_term",
              "range",
              "rank",
              "reconstructed",
              "reduce_max",
              "reduce_min",
              "reduce_sum",
              "residuals",
              "rhs",
              "rule",
              "scale",
              "scaled",
              "score",
              "score_terms",
              "self",
              "shape",
              "solve_weight",
              "solved_t",
              "sqrt",
              "stack",
              "state_covariance",
              "state_factor",
              "state_points",
              "str",
              "symmetrize",
              "tensor",
              "tensordot",
              "tf",
              "tf_srukf_unit_sigma_point_rule",
              "trace",
              "trace_term",
              "transition_fn",
              "transition_jacobian",
              "transition_jacobian_fn",
              "transpose",
              "value",
              "weights",
              "zeros"
            ],
            "code_operations": {
              "names": [
                "Callable",
                "Mapping",
                "MappingProxyType",
                "TFSRUKFMapFn",
                "TFSRUKFParameterDerivativeFn",
                "TFSRUKFPointJacobianFn",
                "TFSRUKFSigmaPointRule",
                "TFSRUKFStepDerivatives",
                "TFSRUKFStepResult",
                "Tensor",
                "ValueError",
                "__setattr__",
                "_as_matrix",
                "_as_vector",
                "_factor_derivative_residual",
                "_factor_from_centered",
                "_factor_reconstruction_residual",
                "_logdet_from_lower_factor",
                "_right_solve_spd",
                "_weighted_covariance",
                "_weighted_covariance_first_derivative",
                "_weighted_mean",
                "abs",
                "append",
                "assert_greater_equal",
                "augmented_factor",
                "augmented_mean",
                "axis_weight",
                "branch_label",
                "cast",
                "centered",
                "centered_left",
                "centered_observation",
                "centered_right",
                "centered_state",
                "cholesky_factor_first_derivatives",
                "concat",
                "constant",
                "control_dependencies",
                "convert_to_tensor",
                "covariance",
                "covariance_factor",
                "covariance_weights",
                "cross_covariance",
                "d_augmented_factor",
                "d_augmented_mean",
                "d_centered_left",
                "d_centered_observation",
                "d_centered_right",
                "d_centered_state",
                "d_covariance",
                "d_cross_covariance",
                "d_factor",
                "d_filtered_covariance",
                "d_filtered_covariance_rows",
                "d_filtered_factor",
                "d_filtered_mean",
                "d_gain",
                "d_gain_rows",
                "d_innovation",
                "d_innovation_covariance",
                "d_observation_direct",
                "d_observation_fn",
                "d_observation_points",
                "d_points",
                "d_predicted_mean",
                "d_predicted_observation",
                "d_s",
                "d_state_covariance",
                "d_state_points",
                "d_transition_direct",
                "d_transition_fn",
                "dataclass",
                "debugging",
                "derivatives",
                "diag_part",
                "diagnostics",
                "dict",
                "dim",
                "einsum",
                "eye",
                "factor",
                "factor_solve",
                "fill",
                "filtered_covariance",
                "filtered_factor",
                "filtered_jitter",
                "filtered_mean",
                "float",
                "float64",
                "gain",
                "getattr",
                "identity",
                "innovation",
                "innovation_covariance",
                "innovation_factor",
                "int",
                "int32",
                "linalg",
                "log",
                "log_likelihood",
                "lower_factor_from_horizontal_stack",
                "math",
                "matrix",
                "matvec",
                "mean_weights",
                "name",
                "newaxis",
                "norm",
                "object",
                "observation",
                "observation_fn",
                "observation_jacobian",
                "observation_jacobian_fn",
                "observation_points",
                "offsets",
                "parameter_dim",
                "parameter_index",
                "pi",
                "point_count",
                "points",
                "predicted_mean",
                "predicted_observation",
                "property",
                "quadratic_term",
                "range",
                "rank",
                "reconstructed",
                "reduce_max",
                "reduce_min",
                "reduce_sum",
                "residuals",
                "rhs",
                "rule",
                "scale",
                "scaled",
                "score",
                "score_terms",
                "self",
                "shape",
                "solve_weight",
                "solved_t",
                "sqrt",
                "stack",
                "state_covariance",
                "state_factor",
                "state_points",
                "str",
                "symmetrize",
                "tensor",
                "tensordot",
                "tf",
                "tf_srukf_unit_sigma_point_rule",
                "trace",
                "trace_term",
                "transition_fn",
                "transition_jacobian",
                "transition_jacobian_fn",
                "transpose",
                "value",
                "weights",
                "zeros"
              ],
              "calls": [
                "MappingProxyType",
                "TFSRUKFSigmaPointRule",
                "TFSRUKFStepResult",
                "ValueError",
                "__setattr__",
                "_as_matrix",
                "_as_vector",
                "_factor_derivative_residual",
                "_factor_from_centered",
                "_factor_reconstruction_residual",
                "_logdet_from_lower_factor",
                "_right_solve_spd",
                "_weighted_covariance",
                "_weighted_covariance_first_derivative",
                "_weighted_mean",
                "abs",
                "append",
                "assert_greater_equal",
                "cast",
                "cholesky_factor_first_derivatives",
                "concat",
                "constant",
                "control_dependencies",
                "convert_to_tensor",
                "d_observation_fn",
                "d_transition_fn",
                "dataclass",
                "diag_part",
                "dict",
                "einsum",
                "eye",
                "factor_solve",
                "fill",
                "getattr",
                "identity",
                "int",
                "log",
                "lower_factor_from_horizontal_stack",
                "matvec",
                "norm",
                "observation_fn",
                "observation_jacobian_fn",
                "range",
                "reduce_max",
                "reduce_min",
                "reduce_sum",
                "shape",
                "sqrt",
                "stack",
                "symmetrize",
                "tensordot",
                "tf_srukf_unit_sigma_point_rule",
                "trace",
                "transition_fn",
                "transition_jacobian_fn",
                "transpose",
                "zeros"
              ],
              "operators": [
                "Add",
                "BitOr",
                "Div",
                "MatMult",
                "Mult",
                "Sub"
              ],
              "function_args": {
                "tf_srukf_unit_sigma_point_rule": [
                  "dim"
                ],
                "_as_vector": [
                  "value"
                ],
                "_as_matrix": [
                  "value"
                ],
                "_weighted_mean": [
                  "points",
                  "weights"
                ],
                "_weighted_covariance": [
                  "centered_left",
                  "centered_right",
                  "weights"
                ],
                "_weighted_covariance_first_derivative": [
                  "centered_left",
                  "centered_right",
                  "d_centered_left",
                  "d_centered_right",
                  "weights"
                ],
                "_factor_from_centered": [
                  "centered",
                  "weights"
                ],
                "_factor_reconstruction_residual": [
                  "factor",
                  "covariance"
                ],
                "_factor_derivative_residual": [
                  "factor",
                  "d_factor",
                  "d_covariance"
                ],
                "_right_solve_spd": [
                  "covariance_factor",
                  "matrix"
                ],
                "_logdet_from_lower_factor": [
                  "factor"
                ],
                "tf_srukf_factor_score_step": [
                  "observation",
                  "augmented_mean",
                  "augmented_factor"
                ],
                "__post_init__": [
                  "self"
                ],
                "point_count": [
                  "self"
                ],
                "parameter_dim": [
                  "self"
                ]
              }
            },
            "boundary": "This trace map records structural term visibility only. It is not semantic proof that the code implements the documented math.",
            "scope_diagnostic": {
              "status": "not_applicable",
              "math_scope_terms": [],
              "mapped_math_scope_terms": [],
              "code_function_args": [
                "augmented_factor",
                "augmented_mean",
                "centered",
                "centered_left",
                "centered_right",
                "covariance",
                "covariance_factor",
                "d_centered_left",
                "d_centered_right",
                "d_covariance",
                "d_factor",
                "dim",
                "factor",
                "matrix",
                "observation",
                "points",
                "self",
                "value",
                "weights"
              ],
              "matched_scope_terms": [],
              "missing_scope_terms": [],
              "function_level_markers": [],
              "supports": "No mathematical function-argument or conditioning boundary was detected for static scope comparison.",
              "does_not_support": "This classification says nothing about semantic code correctness.",
              "safe_wording": "Treat scope comparison as not applicable; retain the ordinary term and conflict diagnostics.",
              "boundary": "Scope diagnostics are structural and non-executing; they are not code correctness proof."
            }
          },
          "conflicts": [],
          "code_summary": {
            "names": [
              "Callable",
              "Mapping",
              "MappingProxyType",
              "TFSRUKFMapFn",
              "TFSRUKFParameterDerivativeFn",
              "TFSRUKFPointJacobianFn",
              "TFSRUKFSigmaPointRule",
              "TFSRUKFStepDerivatives",
              "TFSRUKFStepResult",
              "Tensor",
              "ValueError",
              "__setattr__",
              "_as_matrix",
              "_as_vector",
              "_factor_derivative_residual",
              "_factor_from_centered",
              "_factor_reconstruction_residual",
              "_logdet_from_lower_factor",
              "_right_solve_spd",
              "_weighted_covariance",
              "_weighted_covariance_first_derivative",
              "_weighted_mean",
              "abs",
              "append",
              "assert_greater_equal",
              "augmented_factor",
              "augmented_mean",
              "axis_weight",
              "branch_label",
              "cast",
              "centered",
              "centered_left",
              "centered_observation",
              "centered_right",
              "centered_state",
              "cholesky_factor_first_derivatives",
              "concat",
              "constant",
              "control_dependencies",
              "convert_to_tensor",
              "covariance",
              "covariance_factor",
              "covariance_weights",
              "cross_covariance",
              "d_augmented_factor",
              "d_augmented_mean",
              "d_centered_left",
              "d_centered_observation",
              "d_centered_right",
              "d_centered_state",
              "d_covariance",
              "d_cross_covariance",
              "d_factor",
              "d_filtered_covariance",
              "d_filtered_covariance_rows",
              "d_filtered_factor",
              "d_filtered_mean",
              "d_gain",
              "d_gain_rows",
              "d_innovation",
              "d_innovation_covariance",
              "d_observation_direct",
              "d_observation_fn",
              "d_observation_points",
              "d_points",
              "d_predicted_mean",
              "d_predicted_observation",
              "d_s",
              "d_state_covariance",
              "d_state_points",
              "d_transition_direct",
              "d_transition_fn",
              "dataclass",
              "debugging",
              "derivatives",
              "diag_part",
              "diagnostics",
              "dict",
              "dim",
              "einsum",
              "eye",
              "factor",
              "factor_solve",
              "fill",
              "filtered_covariance",
              "filtered_factor",
              "filtered_jitter",
              "filtered_mean",
              "float",
              "float64",
              "gain",
              "getattr",
              "identity",
              "innovation",
              "innovation_covariance",
              "innovation_factor",
              "int",
              "int32",
              "linalg",
              "log",
              "log_likelihood",
              "lower_factor_from_horizontal_stack",
              "math",
              "matrix",
              "matvec",
              "mean_weights",
              "name",
              "newaxis",
              "norm",
              "object",
              "observation",
              "observation_fn",
              "observation_jacobian",
              "observation_jacobian_fn",
              "observation_points",
              "offsets",
              "parameter_dim",
              "parameter_index",
              "pi",
              "point_count",
              "points",
              "predicted_mean",
              "predicted_observation",
              "property",
              "quadratic_term",
              "range",
              "rank",
              "reconstructed",
              "reduce_max",
              "reduce_min",
              "reduce_sum",
              "residuals",
              "rhs",
              "rule",
              "scale",
              "scaled",
              "score",
              "score_terms",
              "self",
              "shape",
              "solve_weight",
              "solved_t",
              "sqrt",
              "stack",
              "state_covariance",
              "state_factor",
              "state_points",
              "str",
              "symmetrize",
              "tensor",
              "tensordot",
              "tf",
              "tf_srukf_unit_sigma_point_rule",
              "trace",
              "trace_term",
              "transition_fn",
              "transition_jacobian",
              "transition_jacobian_fn",
              "transpose",
              "value",
              "weights",
              "zeros"
            ],
            "calls": [
              "MappingProxyType",
              "TFSRUKFSigmaPointRule",
              "TFSRUKFStepResult",
              "ValueError",
              "__setattr__",
              "_as_matrix",
              "_as_vector",
              "_factor_derivative_residual",
              "_factor_from_centered",
              "_factor_reconstruction_residual",
              "_logdet_from_lower_factor",
              "_right_solve_spd",
              "_weighted_covariance",
              "_weighted_covariance_first_derivative",
              "_weighted_mean",
              "abs",
              "append",
              "assert_greater_equal",
              "cast",
              "cholesky_factor_first_derivatives",
              "concat",
              "constant",
              "control_dependencies",
              "convert_to_tensor",
              "d_observation_fn",
              "d_transition_fn",
              "dataclass",
              "diag_part",
              "dict",
              "einsum",
              "eye",
              "factor_solve",
              "fill",
              "getattr",
              "identity",
              "int",
              "log",
              "lower_factor_from_horizontal_stack",
              "matvec",
              "norm",
              "observation_fn",
              "observation_jacobian_fn",
              "range",
              "reduce_max",
              "reduce_min",
              "reduce_sum",
              "shape",
              "sqrt",
              "stack",
              "symmetrize",
              "tensordot",
              "tf_srukf_unit_sigma_point_rule",
              "trace",
              "transition_fn",
              "transition_jacobian_fn",
              "transpose",
              "zeros"
            ],
            "operators": [
              "Add",
              "BitOr",
              "Div",
              "MatMult",
              "Mult",
              "Sub"
            ],
            "function_args": {
              "tf_srukf_unit_sigma_point_rule": [
                "dim"
              ],
              "_as_vector": [
                "value"
              ],
              "_as_matrix": [
                "value"
              ],
              "_weighted_mean": [
                "points",
                "weights"
              ],
              "_weighted_covariance": [
                "centered_left",
                "centered_right",
                "weights"
              ],
              "_weighted_covariance_first_derivative": [
                "centered_left",
                "centered_right",
                "d_centered_left",
                "d_centered_right",
                "weights"
              ],
              "_factor_from_centered": [
                "centered",
                "weights"
              ],
              "_factor_reconstruction_residual": [
                "factor",
                "covariance"
              ],
              "_factor_derivative_residual": [
                "factor",
                "d_factor",
                "d_covariance"
              ],
              "_right_solve_spd": [
                "covariance_factor",
                "matrix"
              ],
              "_logdet_from_lower_factor": [
                "factor"
              ],
              "tf_srukf_factor_score_step": [
                "observation",
                "augmented_mean",
                "augmented_factor"
              ],
              "__post_init__": [
                "self"
              ],
              "point_count": [
                "self"
              ],
              "parameter_dim": [
                "self"
              ]
            }
          },
          "workbench_result": {
            "question": {
              "question_type": "code_implements_equation",
              "target": "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T",
              "givens": [],
              "assumptions": [],
              "context": {
                "aliases": {}
              },
              "metadata": {
                "schema_version": "1.0",
                "contract": "math_debugging_question"
              }
            },
            "status": "refuted",
            "reason": "Code is missing required equation terms or has structural conflicts.",
            "obligations": [],
            "assumptions": [],
            "backend_attempts": [],
            "counterexamples": [],
            "actions": [
              {
                "kind": "inspect_missing_or_conflicting_code_terms"
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
            "contract": "equation_code_match_result"
          }
        }
      }
    ],
    "assumption_items": [],
    "veto_items": [],
    "action_items": [
      {
        "code": "human_review",
        "description": "Inspect structural mismatch or scope limitation."
      },
      {
        "code": "human_review",
        "description": "Review structural matches, missing terms, aliases, and audit-only extras before treating code as correct."
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
        "code": "structural_evidence_not_proof",
        "text": "Structural evidence is not a semantic proof."
      }
    ],
    "non_claim_codes": [
      "general_theorem_proving_not_claimed",
      "release_readiness_not_claimed",
      "structural_evidence_not_proof"
    ],
    "boundary": "This ledger is case-local provenance for the same high-level workflow envelope. It is not independent proof, release evidence, public benchmark validation, or a claim of broad downstream-agent usefulness."
  },
  "metadata": {
    "schema_version": "1.0",
    "contract": "high_level_workflow_result"
  }
}
=== C3: code-implements-equation logdet vs srukf_factor_tf.py ===
{
  "status": "mismatch",
  "reason": "Code is missing required equation terms or has structural conflicts.",
  "equation": "logdet(P_y) = 2 sum(log(diag(S_y)))",
  "matched_terms": [
    "log"
  ],
  "missing_terms": [
    "P_y",
    "S_y",
    "diag",
    "logdet",
    "sum"
  ],
  "extra_code_terms": [
    "Callable",
    "Mapping",
    "MappingProxyType",
    "TFSRUKFMapFn",
    "TFSRUKFParameterDerivativeFn",
    "TFSRUKFPointJacobianFn",
    "TFSRUKFSigmaPointRule",
    "TFSRUKFStepDerivatives",
    "TFSRUKFStepResult",
    "Tensor",
    "ValueError",
    "__setattr__",
    "_as_matrix",
    "_as_vector",
    "_factor_derivative_residual",
    "_factor_from_centered",
    "_factor_reconstruction_residual",
    "_logdet_from_lower_factor",
    "_right_solve_spd",
    "_weighted_covariance",
    "_weighted_covariance_first_derivative",
    "_weighted_mean",
    "abs",
    "append",
    "assert_greater_equal",
    "augmented_factor",
    "augmented_mean",
    "axis_weight",
    "branch_label",
    "cast",
    "centered",
    "centered_left",
    "centered_observation",
    "centered_right",
    "centered_state",
    "cholesky_factor_first_derivatives",
    "concat",
    "constant",
    "control_dependencies",
    "convert_to_tensor",
    "covariance",
    "covariance_factor",
    "covariance_weights",
    "cross_covariance",
    "d_augmented_factor",
    "d_augmented_mean",
    "d_centered_left",
    "d_centered_observation",
    "d_centered_right",
    "d_centered_state",
    "d_covariance",
    "d_cross_covariance",
    "d_factor",
    "d_filtered_covariance",
    "d_filtered_covariance_rows",
    "d_filtered_factor",
    "d_filtered_mean",
    "d_gain",
    "d_gain_rows",
    "d_innovation",
    "d_innovation_covariance",
    "d_observation_direct",
    "d_observation_fn",
    "d_observation_points",
    "d_points",
    "d_predicted_mean",
    "d_predicted_observation",
    "d_s",
    "d_state_covariance",
    "d_state_points",
    "d_transition_direct",
    "d_transition_fn",
    "dataclass",
    "debugging",
    "derivatives",
    "diag_part",
    "diagnostics",
    "dict",
    "dim",
    "einsum",
    "eye",
    "factor",
    "factor_solve",
    "fill",
    "filtered_covariance",
    "filtered_factor",
    "filtered_jitter",
    "filtered_mean",
    "float",
    "float64",
    "gain",
    "getattr",
    "identity",
    "innovation",
    "innovation_covariance",
    "innovation_factor",
    "int",
    "int32",
    "linalg",
    "log_likelihood",
    "lower_factor_from_horizontal_stack",
    "math",
    "matrix",
    "matvec",
    "mean_weights",
    "name",
    "newaxis",
    "norm",
    "object",
    "observation",
    "observation_fn",
    "observation_jacobian",
    "observation_jacobian_fn",
    "observation_points",
    "offsets",
    "parameter_dim",
    "parameter_index",
    "pi",
    "point_count",
    "points",
    "predicted_mean",
    "predicted_observation",
    "property",
    "quadratic_term",
    "range",
    "rank",
    "reconstructed",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    "residuals",
    "rhs",
    "rule",
    "scale",
    "scaled",
    "score",
    "score_terms",
    "self",
    "shape",
    "solve_weight",
    "solved_t",
    "sqrt",
    "stack",
    "state_covariance",
    "state_factor",
    "state_points",
    "str",
    "symmetrize",
    "tensor",
    "tensordot",
    "tf",
    "tf_srukf_unit_sigma_point_rule",
    "trace",
    "trace_term",
    "transition_fn",
    "transition_jacobian",
    "transition_jacobian_fn",
    "transpose",
    "value",
    "weights",
    "zeros"
  ],
  "trace_map": {
    "equation_terms": [
      "P_y",
      "S_y",
      "diag",
      "log",
      "logdet",
      "sum"
    ],
    "alias_map": {},
    "mapped_terms": [
      "P_y",
      "S_y",
      "diag",
      "log",
      "logdet",
      "sum"
    ],
    "term_traces": [
      {
        "equation_term": "P_y",
        "mapped_code_term": "P_y",
        "matched": false,
        "source": "direct"
      },
      {
        "equation_term": "S_y",
        "mapped_code_term": "S_y",
        "matched": false,
        "source": "direct"
      },
      {
        "equation_term": "diag",
        "mapped_code_term": "diag",
        "matched": false,
        "source": "direct"
      },
      {
        "equation_term": "log",
        "mapped_code_term": "log",
        "matched": true,
        "source": "direct"
      },
      {
        "equation_term": "logdet",
        "mapped_code_term": "logdet",
        "matched": false,
        "source": "direct"
      },
      {
        "equation_term": "sum",
        "mapped_code_term": "sum",
        "matched": false,
        "source": "direct"
      }
    ],
    "alias_collisions": [],
    "matched_terms": [
      "log"
    ],
    "missing_terms": [
      "P_y",
      "S_y",
      "diag",
      "logdet",
      "sum"
    ],
    "extra_code_terms": [
      "Callable",
      "Mapping",
      "MappingProxyType",
      "TFSRUKFMapFn",
      "TFSRUKFParameterDerivativeFn",
      "TFSRUKFPointJacobianFn",
      "TFSRUKFSigmaPointRule",
      "TFSRUKFStepDerivatives",
      "TFSRUKFStepResult",
      "Tensor",
      "ValueError",
      "__setattr__",
      "_as_matrix",
      "_as_vector",
      "_factor_derivative_residual",
      "_factor_from_centered",
      "_factor_reconstruction_residual",
      "_logdet_from_lower_factor",
      "_right_solve_spd",
      "_weighted_covariance",
      "_weighted_covariance_first_derivative",
      "_weighted_mean",
      "abs",
      "append",
      "assert_greater_equal",
      "augmented_factor",
      "augmented_mean",
      "axis_weight",
      "branch_label",
      "cast",
      "centered",
      "centered_left",
      "centered_observation",
      "centered_right",
      "centered_state",
      "cholesky_factor_first_derivatives",
      "concat",
      "constant",
      "control_dependencies",
      "convert_to_tensor",
      "covariance",
      "covariance_factor",
      "covariance_weights",
      "cross_covariance",
      "d_augmented_factor",
      "d_augmented_mean",
      "d_centered_left",
      "d_centered_observation",
      "d_centered_right",
      "d_centered_state",
      "d_covariance",
      "d_cross_covariance",
      "d_factor",
      "d_filtered_covariance",
      "d_filtered_covariance_rows",
      "d_filtered_factor",
      "d_filtered_mean",
      "d_gain",
      "d_gain_rows",
      "d_innovation",
      "d_innovation_covariance",
      "d_observation_direct",
      "d_observation_fn",
      "d_observation_points",
      "d_points",
      "d_predicted_mean",
      "d_predicted_observation",
      "d_s",
      "d_state_covariance",
      "d_state_points",
      "d_transition_direct",
      "d_transition_fn",
      "dataclass",
      "debugging",
      "derivatives",
      "diag_part",
      "diagnostics",
      "dict",
      "dim",
      "einsum",
      "eye",
      "factor",
      "factor_solve",
      "fill",
      "filtered_covariance",
      "filtered_factor",
      "filtered_jitter",
      "filtered_mean",
      "float",
      "float64",
      "gain",
      "getattr",
      "identity",
      "innovation",
      "innovation_covariance",
      "innovation_factor",
      "int",
      "int32",
      "linalg",
      "log_likelihood",
      "lower_factor_from_horizontal_stack",
      "math",
      "matrix",
      "matvec",
      "mean_weights",
      "name",
      "newaxis",
      "norm",
      "object",
      "observation",
      "observation_fn",
      "observation_jacobian",
      "observation_jacobian_fn",
      "observation_points",
      "offsets",
      "parameter_dim",
      "parameter_index",
      "pi",
      "point_count",
      "points",
      "predicted_mean",
      "predicted_observation",
      "property",
      "quadratic_term",
      "range",
      "rank",
      "reconstructed",
      "reduce_max",
      "reduce_min",
      "reduce_sum",
      "residuals",
      "rhs",
      "rule",
      "scale",
      "scaled",
      "score",
      "score_terms",
      "self",
      "shape",
      "solve_weight",
      "solved_t",
      "sqrt",
      "stack",
      "state_covariance",
      "state_factor",
      "state_points",
      "str",
      "symmetrize",
      "tensor",
      "tensordot",
      "tf",
      "tf_srukf_unit_sigma_point_rule",
      "trace",
      "trace_term",
      "transition_fn",
      "transition_jacobian",
      "transition_jacobian_fn",
      "transpose",
      "value",
      "weights",
      "zeros"
    ],
    "code_operations": {
      "names": [
        "Callable",
        "Mapping",
        "MappingProxyType",
        "TFSRUKFMapFn",
        "TFSRUKFParameterDerivativeFn",
        "TFSRUKFPointJacobianFn",
        "TFSRUKFSigmaPointRule",
        "TFSRUKFStepDerivatives",
        "TFSRUKFStepResult",
        "Tensor",
        "ValueError",
        "__setattr__",
        "_as_matrix",
        "_as_vector",
        "_factor_derivative_residual",
        "_factor_from_centered",
        "_factor_reconstruction_residual",
        "_logdet_from_lower_factor",
        "_right_solve_spd",
        "_weighted_covariance",
        "_weighted_covariance_first_derivative",
        "_weighted_mean",
        "abs",
        "append",
        "assert_greater_equal",
        "augmented_factor",
        "augmented_mean",
        "axis_weight",
        "branch_label",
        "cast",
        "centered",
        "centered_left",
        "centered_observation",
        "centered_right",
        "centered_state",
        "cholesky_factor_first_derivatives",
        "concat",
        "constant",
        "control_dependencies",
        "convert_to_tensor",
        "covariance",
        "covariance_factor",
        "covariance_weights",
        "cross_covariance",
        "d_augmented_factor",
        "d_augmented_mean",
        "d_centered_left",
        "d_centered_observation",
        "d_centered_right",
        "d_centered_state",
        "d_covariance",
        "d_cross_covariance",
        "d_factor",
        "d_filtered_covariance",
        "d_filtered_covariance_rows",
        "d_filtered_factor",
        "d_filtered_mean",
        "d_gain",
        "d_gain_rows",
        "d_innovation",
        "d_innovation_covariance",
        "d_observation_direct",
        "d_observation_fn",
        "d_observation_points",
        "d_points",
        "d_predicted_mean",
        "d_predicted_observation",
        "d_s",
        "d_state_covariance",
        "d_state_points",
        "d_transition_direct",
        "d_transition_fn",
        "dataclass",
        "debugging",
        "derivatives",
        "diag_part",
        "diagnostics",
        "dict",
        "dim",
        "einsum",
        "eye",
        "factor",
        "factor_solve",
        "fill",
        "filtered_covariance",
        "filtered_factor",
        "filtered_jitter",
        "filtered_mean",
        "float",
        "float64",
        "gain",
        "getattr",
        "identity",
        "innovation",
        "innovation_covariance",
        "innovation_factor",
        "int",
        "int32",
        "linalg",
        "log",
        "log_likelihood",
        "lower_factor_from_horizontal_stack",
        "math",
        "matrix",
        "matvec",
        "mean_weights",
        "name",
        "newaxis",
        "norm",
        "object",
        "observation",
        "observation_fn",
        "observation_jacobian",
        "observation_jacobian_fn",
        "observation_points",
        "offsets",
        "parameter_dim",
        "parameter_index",
        "pi",
        "point_count",
        "points",
        "predicted_mean",
        "predicted_observation",
        "property",
        "quadratic_term",
        "range",
        "rank",
        "reconstructed",
        "reduce_max",
        "reduce_min",
        "reduce_sum",
        "residuals",
        "rhs",
        "rule",
        "scale",
        "scaled",
        "score",
        "score_terms",
        "self",
        "shape",
        "solve_weight",
        "solved_t",
        "sqrt",
        "stack",
        "state_covariance",
        "state_factor",
        "state_points",
        "str",
        "symmetrize",
        "tensor",
        "tensordot",
        "tf",
        "tf_srukf_unit_sigma_point_rule",
        "trace",
        "trace_term",
        "transition_fn",
        "transition_jacobian",
        "transition_jacobian_fn",
        "transpose",
        "value",
        "weights",
        "zeros"
      ],
      "calls": [
        "MappingProxyType",
        "TFSRUKFSigmaPointRule",
        "TFSRUKFStepResult",
        "ValueError",
        "__setattr__",
        "_as_matrix",
        "_as_vector",
        "_factor_derivative_residual",
        "_factor_from_centered",
        "_factor_reconstruction_residual",
        "_logdet_from_lower_factor",
        "_right_solve_spd",
        "_weighted_covariance",
        "_weighted_covariance_first_derivative",
        "_weighted_mean",
        "abs",
        "append",
        "assert_greater_equal",
        "cast",
        "cholesky_factor_first_derivatives",
        "concat",
        "constant",
        "control_dependencies",
        "convert_to_tensor",
        "d_observation_fn",
        "d_transition_fn",
        "dataclass",
        "diag_part",
        "dict",
        "einsum",
        "eye",
        "factor_solve",
        "fill",
        "getattr",
        "identity",
        "int",
        "log",
        "lower_factor_from_horizontal_stack",
        "matvec",
        "norm",
        "observation_fn",
        "observation_jacobian_fn",
        "range",
        "reduce_max",
        "reduce_min",
        "reduce_sum",
        "shape",
        "sqrt",
        "stack",
        "symmetrize",
        "tensordot",
        "tf_srukf_unit_sigma_point_rule",
        "trace",
        "transition_fn",
        "transition_jacobian_fn",
        "transpose",
        "zeros"
      ],
      "operators": [
        "Add",
        "BitOr",
        "Div",
        "MatMult",
        "Mult",
        "Sub"
      ],
      "function_args": {
        "tf_srukf_unit_sigma_point_rule": [
          "dim"
        ],
        "_as_vector": [
          "value"
        ],
        "_as_matrix": [
          "value"
        ],
        "_weighted_mean": [
          "points",
          "weights"
        ],
        "_weighted_covariance": [
          "centered_left",
          "centered_right",
          "weights"
        ],
        "_weighted_covariance_first_derivative": [
          "centered_left",
          "centered_right",
          "d_centered_left",
          "d_centered_right",
          "weights"
        ],
        "_factor_from_centered": [
          "centered",
          "weights"
        ],
        "_factor_reconstruction_residual": [
          "factor",
          "covariance"
        ],
        "_factor_derivative_residual": [
          "factor",
          "d_factor",
          "d_covariance"
        ],
        "_right_solve_spd": [
          "covariance_factor",
          "matrix"
        ],
        "_logdet_from_lower_factor": [
          "factor"
        ],
        "tf_srukf_factor_score_step": [
          "observation",
          "augmented_mean",
          "augmented_factor"
        ],
        "__post_init__": [
          "self"
        ],
        "point_count": [
          "self"
        ],
        "parameter_dim": [
          "self"
        ]
      }
    },
    "boundary": "This trace map records structural term visibility only. It is not semantic proof that the code implements the documented math.",
    "scope_diagnostic": {
      "status": "scope_limited",
      "math_scope_terms": [
        "P_y",
        "log(diag(S_y"
      ],
      "mapped_math_scope_terms": [
        "P_y",
        "log(diag(S_y"
      ],
      "code_function_args": [
        "augmented_factor",
        "augmented_mean",
        "centered",
        "centered_left",
        "centered_right",
        "covariance",
        "covariance_factor",
        "d_centered_left",
        "d_centered_right",
        "d_covariance",
        "d_factor",
        "dim",
        "factor",
        "matrix",
        "observation",
        "points",
        "self",
        "value",
        "weights"
      ],
      "matched_scope_terms": [],
      "missing_scope_terms": [
        "P_y",
        "log(diag(S_y"
      ],
      "function_level_markers": [
        "P_y",
        "log(diag(S_y"
      ],
      "supports": "The code exposes structurally relevant terms for a value-level or instance-level slice, with a callable signature covering only part of the mathematical function-level scope.",
      "does_not_support": "The code signature omits one or more function-level arguments, summation/index domains, or conditioning-scope markers required by the mathematical claim.",
      "safe_wording": "Treat this as scope-limited implementation evidence: it may support one evaluated slice, but it does not prove the full function-level formula.",
      "boundary": "Scope diagnostics are structural and non-executing; they are not code correctness proof."
    }
  },
  "conflicts": [],
  "code_summary": {
    "names": [
      "Callable",
      "Mapping",
      "MappingProxyType",
      "TFSRUKFMapFn",
      "TFSRUKFParameterDerivativeFn",
      "TFSRUKFPointJacobianFn",
      "TFSRUKFSigmaPointRule",
      "TFSRUKFStepDerivatives",
      "TFSRUKFStepResult",
      "Tensor",
      "ValueError",
      "__setattr__",
      "_as_matrix",
      "_as_vector",
      "_factor_derivative_residual",
      "_factor_from_centered",
      "_factor_reconstruction_residual",
      "_logdet_from_lower_factor",
      "_right_solve_spd",
      "_weighted_covariance",
      "_weighted_covariance_first_derivative",
      "_weighted_mean",
      "abs",
      "append",
      "assert_greater_equal",
      "augmented_factor",
      "augmented_mean",
      "axis_weight",
      "branch_label",
      "cast",
      "centered",
      "centered_left",
      "centered_observation",
      "centered_right",
      "centered_state",
      "cholesky_factor_first_derivatives",
      "concat",
      "constant",
      "control_dependencies",
      "convert_to_tensor",
      "covariance",
      "covariance_factor",
      "covariance_weights",
      "cross_covariance",
      "d_augmented_factor",
      "d_augmented_mean",
      "d_centered_left",
      "d_centered_observation",
      "d_centered_right",
      "d_centered_state",
      "d_covariance",
      "d_cross_covariance",
      "d_factor",
      "d_filtered_covariance",
      "d_filtered_covariance_rows",
      "d_filtered_factor",
      "d_filtered_mean",
      "d_gain",
      "d_gain_rows",
      "d_innovation",
      "d_innovation_covariance",
      "d_observation_direct",
      "d_observation_fn",
      "d_observation_points",
      "d_points",
      "d_predicted_mean",
      "d_predicted_observation",
      "d_s",
      "d_state_covariance",
      "d_state_points",
      "d_transition_direct",
      "d_transition_fn",
      "dataclass",
      "debugging",
      "derivatives",
      "diag_part",
      "diagnostics",
      "dict",
      "dim",
      "einsum",
      "eye",
      "factor",
      "factor_solve",
      "fill",
      "filtered_covariance",
      "filtered_factor",
      "filtered_jitter",
      "filtered_mean",
      "float",
      "float64",
      "gain",
      "getattr",
      "identity",
      "innovation",
      "innovation_covariance",
      "innovation_factor",
      "int",
      "int32",
      "linalg",
      "log",
      "log_likelihood",
      "lower_factor_from_horizontal_stack",
      "math",
      "matrix",
      "matvec",
      "mean_weights",
      "name",
      "newaxis",
      "norm",
      "object",
      "observation",
      "observation_fn",
      "observation_jacobian",
      "observation_jacobian_fn",
      "observation_points",
      "offsets",
      "parameter_dim",
      "parameter_index",
      "pi",
      "point_count",
      "points",
      "predicted_mean",
      "predicted_observation",
      "property",
      "quadratic_term",
      "range",
      "rank",
      "reconstructed",
      "reduce_max",
      "reduce_min",
      "reduce_sum",
      "residuals",
      "rhs",
      "rule",
      "scale",
      "scaled",
      "score",
      "score_terms",
      "self",
      "shape",
      "solve_weight",
      "solved_t",
      "sqrt",
      "stack",
      "state_covariance",
      "state_factor",
      "state_points",
      "str",
      "symmetrize",
      "tensor",
      "tensordot",
      "tf",
      "tf_srukf_unit_sigma_point_rule",
      "trace",
      "trace_term",
      "transition_fn",
      "transition_jacobian",
      "transition_jacobian_fn",
      "transpose",
      "value",
      "weights",
      "zeros"
    ],
    "calls": [
      "MappingProxyType",
      "TFSRUKFSigmaPointRule",
      "TFSRUKFStepResult",
      "ValueError",
      "__setattr__",
      "_as_matrix",
      "_as_vector",
      "_factor_derivative_residual",
      "_factor_from_centered",
      "_factor_reconstruction_residual",
      "_logdet_from_lower_factor",
      "_right_solve_spd",
      "_weighted_covariance",
      "_weighted_covariance_first_derivative",
      "_weighted_mean",
      "abs",
      "append",
      "assert_greater_equal",
      "cast",
      "cholesky_factor_first_derivatives",
      "concat",
      "constant",
      "control_dependencies",
      "convert_to_tensor",
      "d_observation_fn",
      "d_transition_fn",
      "dataclass",
      "diag_part",
      "dict",
      "einsum",
      "eye",
      "factor_solve",
      "fill",
      "getattr",
      "identity",
      "int",
      "log",
      "lower_factor_from_horizontal_stack",
      "matvec",
      "norm",
      "observation_fn",
      "observation_jacobian_fn",
      "range",
      "reduce_max",
      "reduce_min",
      "reduce_sum",
      "shape",
      "sqrt",
      "stack",
      "symmetrize",
      "tensordot",
      "tf_srukf_unit_sigma_point_rule",
      "trace",
      "transition_fn",
      "transition_jacobian_fn",
      "transpose",
      "zeros"
    ],
    "operators": [
      "Add",
      "BitOr",
      "Div",
      "MatMult",
      "Mult",
      "Sub"
    ],
    "function_args": {
      "tf_srukf_unit_sigma_point_rule": [
        "dim"
      ],
      "_as_vector": [
        "value"
      ],
      "_as_matrix": [
        "value"
      ],
      "_weighted_mean": [
        "points",
        "weights"
      ],
      "_weighted_covariance": [
        "centered_left",
        "centered_right",
        "weights"
      ],
      "_weighted_covariance_first_derivative": [
        "centered_left",
        "centered_right",
        "d_centered_left",
        "d_centered_right",
        "weights"
      ],
      "_factor_from_centered": [
        "centered",
        "weights"
      ],
      "_factor_reconstruction_residual": [
        "factor",
        "covariance"
      ],
      "_factor_derivative_residual": [
        "factor",
        "d_factor",
        "d_covariance"
      ],
      "_right_solve_spd": [
        "covariance_factor",
        "matrix"
      ],
      "_logdet_from_lower_factor": [
        "factor"
      ],
      "tf_srukf_factor_score_step": [
        "observation",
        "augmented_mean",
        "augmented_factor"
      ],
      "__post_init__": [
        "self"
      ],
      "point_count": [
        "self"
      ],
      "parameter_dim": [
        "self"
      ]
    }
  },
  "workbench_result": {
    "question": {
      "question_type": "code_implements_equation",
      "target": "logdet(P_y) = 2 sum(log(diag(S_y)))",
      "givens": [],
      "assumptions": [],
      "context": {
        "aliases": {}
      },
      "metadata": {
        "schema_version": "1.0",
        "contract": "math_debugging_question"
      }
    },
    "status": "refuted",
    "reason": "Code is missing required equation terms or has structural conflicts.",
    "obligations": [],
    "assumptions": [],
    "backend_attempts": [],
    "counterexamples": [],
    "actions": [
      {
        "kind": "inspect_missing_or_conflicting_code_terms"
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
    "contract": "equation_code_match_result"
  }
}
