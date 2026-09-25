"""Persistent paired-loss prefixes for immutable q20 map checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_training_protocol import HeldoutLoss
from bayesfilter.inference.q20_production_config import digest, write_json
from bayesfilter.inference.tempered_transport_ensemble_tf import restore_trainable_transport_checkpoint


class ValidationBudgetExhausted(RuntimeError):
    """Completed validation blocks are retained for unchanged-scope continuation."""


class FrozenLossCache:
    def __init__(self, root, *, bridge, scope, batch_size, jit_compile):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bridge, self.scope = bridge, scope
        self.batch_size, self.jit_compile = batch_size, jit_compile
        self.entries = {}
        self.evaluated_rows = 0
        self.reused_rows = 0

    def _entry(self, map_state, seed):
        identity = {"schema": "bayesfilter.q20.frozen_loss_prefix.v1",
            "map": map_state, "seed": list(seed), "scope": self.scope,
            "bridge_signature": self.bridge.signature, "target_signature": self.bridge.target_signature,
            "beta": map_state["beta"], "batch_size": self.batch_size,
            "jit_compile": self.jit_compile, "dtype": "float64"}
        identity = json.loads(json.dumps(identity, allow_nan=False))
        key = digest(identity)
        if key in self.entries:
            return self.entries[key]
        # The restored object belongs only to this evaluator, never an optimizer.
        frozen = restore_trainable_transport_checkpoint(map_state, expected_context={
            "bridge_signature": self.bridge.signature, "target_signature": self.bridge.target_signature})
        folder = self.root / key
        folder.mkdir(exist_ok=True)
        entry = {"identity": identity, "folder": folder, "transport": frozen,
                 "loss": tf.zeros([0], tf.float64),
                 "evaluator": HeldoutLoss(frozen, self.bridge, map_state["beta"],
                     batch_size=self.batch_size, jit_compile=self.jit_compile)}
        receipt = folder / "prefix.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["identity"] != identity:
                raise ValueError("heldout cache identity changed")
            rows = record["rows"]
            if type(rows) is not int or rows <= 0 or rows % self.batch_size:
                raise ValueError("heldout cache row count invalid")
            data = (folder / f"loss-{rows:08d}.tensor").read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("heldout cache checksum mismatch")
            loss = tf.io.parse_tensor(data, out_type=tf.float64)
            if loss.shape != (rows,):
                raise ValueError("heldout cache shape mismatch")
            tf.debugging.assert_all_finite(loss, "heldout cache")
            entry["loss"] = loss
        self.entries[key] = entry
        return entry

    def post_training_probe(self, map_state, *, rows=1000, batch_size=None, seed, budget_check=None):
        """Persist every geometry batch; incomplete prefixes cannot be admitted."""
        from bayesfilter.inference import neutra_post_training as diagnostic
        import math
        batch_size = math.gcd(rows, diagnostic.POST_TRAINING_BATCH_SIZE) if batch_size is None else batch_size
        entry = self._entry(map_state, seed)
        identity = {"schema": diagnostic.PROBE_SCHEMA, "evaluation": entry["identity"],
                    "rows": rows, "batch_size": batch_size, "seed": list(seed),
                    "diagnostic_source_sha256": hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest()}
        path = entry["folder"] / f"geometry-{digest(identity)}.json"
        folder = path.with_suffix("")
        folder.mkdir(exist_ok=True)

        def read_block(index):
            block_path = folder / f"batch-{index:05d}.json"
            block = json.loads(block_path.read_text())
            if (block["identity"] != digest(identity) or block["index"] != index
                    or block["checksum"] != digest(block["result"])):
                raise ValueError("post-training probe batch identity/checksum mismatch")
            return block["result"]

        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["identity"] != identity or saved["checksum"] != digest(saved["result"])
                    or saved["result"].get("schema") != diagnostic.PROBE_SCHEMA):
                raise ValueError("post-training probe cache identity/checksum mismatch")
            for index in range(rows // batch_size):
                read_block(index)
            return {**saved["result"], "cache_reused": True, "points_directory": str(folder)}
        probe = diagnostic.PostTrainingProbe(entry["transport"], self.bridge, map_state["beta"],
            rows=rows, batch_size=batch_size, jit_compile=self.jit_compile)
        latent = probe.latent_bank(seed)
        blocks = []
        for index, start in enumerate(range(0, rows, batch_size)):
            block_path = folder / f"batch-{index:05d}.json"
            if block_path.exists():
                blocks.append(read_block(index))
                continue
            if budget_check is not None and not budget_check():
                raise ValidationBudgetExhausted(
                    f"post-training probe paused after {len(blocks)*batch_size}/{rows} rows")
            block = probe.batch(latent[start:start+batch_size])
            write_json(block_path, {"identity": digest(identity), "index": index,
                "result": block, "checksum": digest(block)})
            blocks.append(block)
        result = probe.summarize(blocks, seed)
        write_json(path, {"identity": identity, "result": result, "checksum": digest(result)})
        return {**result, "cache_reused": False, "points_directory": str(folder)}

    def _save(self, entry):
        rows = int(entry["loss"].shape[0])
        if not rows:
            return
        data = tf.io.serialize_tensor(entry["loss"]).numpy()
        path = entry["folder"] / f"loss-{rows:08d}.tensor"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("heldout cache prefix changed")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        write_json(entry["folder"] / "prefix.json", {"identity": entry["identity"],
            "rows": rows, "sha256": hashlib.sha256(data).hexdigest()}, exclusive=False)

    def evaluate(self, map_state, count, seed, *, budget_check=None):
        if type(count) is not int or count <= 0 or count % self.batch_size:
            raise ValueError("cached validation requires whole positive batches")
        entry = self._entry(map_state, seed)
        existing = int(entry["loss"].shape[0])
        self.reused_rows += min(existing, count)
        blocks = [entry["loss"]]
        try:
            for start in range(existing, count, self.batch_size):
                if budget_check is not None and not budget_check():
                    raise ValidationBudgetExhausted("heldout validation paused at batch boundary")
                loss, valid = entry["evaluator"].batch(start // self.batch_size, seed)
                if not bool(tf.reduce_all(valid).numpy()):
                    raise ValueError("invalid heldout target/map row; no resampling allowed")
                tf.debugging.assert_all_finite(loss, "heldout losses")
                blocks.append(loss)
                self.evaluated_rows += self.batch_size
        finally:
            if len(blocks) > 1:
                entry["loss"] = tf.concat(blocks, axis=0)
                self._save(entry)
        return entry["loss"][:count]
