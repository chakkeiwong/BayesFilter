"""Budgeted master refresh: checked continuation import and separate cost stages."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from bayesfilter.inference.q20_campaign_runtime import atomic_json
from bayesfilter.inference.q20_training_resume import checksum, import_training_checkpoint, read_training_checkpoint


PLAN = "docs/plans/bayesfilter-ssl-lstm-q20-master-resume-2026-09-18.md"
PHASE_CAP = 14400.


class RefreshIncomplete(RuntimeError):
    def __init__(self, name, result):
        self.name, self.result = name, result


def refresh_master(campaign, *, checkpoint=None, previous_source_root=None, fixture=False):
    from bayesfilter.inference.q20_production_config import digest
    from bayesfilter.inference.q20_campaign_costs import training_reservation
    from bayesfilter.inference.q20_master_program import forecast_campaign
    config = campaign.config
    method = "neutra"
    saved = campaign.state.get("training_resume")
    if saved is None:
        if checkpoint is None or previous_source_root is None:
            raise ValueError("refresh requires a completed training checkpoint and its preserved source root")
        saved = import_training_checkpoint(checkpoint, config, campaign.root / "training-import",
            previous_root=previous_source_root, current_root=campaign.repo)
        campaign.state["training_resume"] = saved
        campaign.save()
    else:
        if checkpoint is not None and str(Path(checkpoint).resolve()) != saved["source_import"]["previous_checkpoint"]:
            raise ValueError("refresh input checkpoint changed")
        if checksum(saved["checkpoint"]) != saved["sha256"]:
            raise ValueError("imported checkpoint changed")
        read_training_checkpoint(saved["checkpoint"], config, sources=campaign.state["sources"])
    qualification = None
    prefix = "refresh-"

    def remaining():
        spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"].startswith(prefix))
        return min(campaign.remaining(True), PHASE_CAP-spent)

    def finish(status, **details):
        campaign.state.update(status=status, details=details)
        campaign.save()
        result = {"status": status, "details": details, "training_resume": saved,
            "remaining_campaign_seconds": campaign.remaining(), "remaining_diagnostic_seconds": campaign.remaining(True),
            "production_qualified": False, "posterior_qualified": False}
        atomic_json(campaign.root / "result.json", result)
        atomic_json(campaign.root / "settled-allowance.json", {
            "schema": "bayesfilter.q20.recovered_allowance.v1", "source_ledger": str(campaign.path),
            "source_sha256": checksum(campaign.path), "campaign_remaining_seconds": campaign.remaining(),
            "diagnostic_remaining_seconds": campaign.remaining(True), "all_attempts_settled": True,
            "master_status": status, "scope": "Diagnostics included in campaign; no renewed allowance."})
        return result

    def stage(name, request, cap):
        name = prefix + name
        request = {**request, "fixture": fixture, "plan_file": PLAN}
        if qualification:
            request["qualification_path"] = qualification
        if name not in campaign.state["stages"]:
            spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
            cap = min(cap-spent, remaining())
            if cap <= 2*config["execution"]["termination_grace_seconds"]:
                raise RefreshIncomplete(name, {"status": "phase_budget_exhausted"})
        result = campaign.numerical_stage(name, request, cap_seconds=cap, diagnostic=True, gpu="auto")
        if not result.get("completed"):
            raise RefreshIncomplete(name, result)
        return result

    try:
        if config["jit_compile"]:
            records = []
            for beta in (1.,):
                result = stage(f"qualify-beta{beta:g}", {"stage": "qualify", "betas": [beta]}, 1200.)
                receipt = json.loads(Path(result["result_path"]).read_text())
                if receipt.pop("checksum") != digest(receipt):
                    raise ValueError("qualification receipt checksum mismatch")
                records.append(receipt)
            combined = {**records[0], "betas": {}}
            for receipt in records:
                if any(receipt[k] != combined[k] for k in receipt if k != "betas"):
                    raise ValueError("qualification scopes differ")
                combined["betas"].update(receipt["betas"])
            qualification = str(campaign.root / "qualification.json")
            atomic_json(qualification, {**combined, "checksum": digest(combined)})
        # Refresh the next authorized estimation method. Classical preparation
        # is not part of plain NeuTra or its frozen identity-mass chart.
        priced = stage("price-training", {"stage": "price-training", "method": method,
            "training_checkpoint": saved["checkpoint"]}, 1200.)
        quote = training_reservation(config, priced["result"]["rows"],
            checkpoint=saved["checkpoint"], method=method)
        atomic_json(campaign.root / "remaining-training-forecast.json", quote)
        downstream = stage("price-downstream", {"stage": "price-downstream", "method": method, "training_checkpoint": saved["checkpoint"],
            "training_pricing": priced["result_path"]}, 6000.)
        prices = downstream["result"]
        prices["missing_cost_categories"] = list(prices.get("missing_cost_categories", []))
        prices["process_overhead_seconds"] = max(0., downstream["supervisor_seconds"]-downstream["wall_seconds"])
        atomic_json(campaign.root / "remaining-prices.json", prices)
        forecast = forecast_campaign(config, prices)
        atomic_json(campaign.root / "forecast.json", forecast)
        return finish("MASTER_REFRESH_PRICING_INCOMPLETE" if prices["missing_cost_categories"] else "MASTER_REFRESH_COMPLETE",
            training_quote=quote, forecast=forecast, missing_cost_categories=prices["missing_cost_categories"],
            method=method,
            next_action="resume the priced plain-NeuTra estimation phases within the campaign allowance")
    except RefreshIncomplete as error:
        waiting = error.result.get("status") == "waiting_for_gpu"
        return finish("WAITING_FOR_GPU" if waiting else "MASTER_REFRESH_PAUSED",
            incomplete_stage=error.name, result=error.result,
            next_action="resume unchanged master within the shared phase cap; completed stages and checkpoints are preserved")
