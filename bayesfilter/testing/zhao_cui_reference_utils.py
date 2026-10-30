"""Reference-only utilities for auditing the pinned Zhao--Cui MATLAB source.

This module is intentionally under bayesfilter.testing. It is not imported
by a production filtering or training route. It uses only Python's standard
library so that source extraction and scalar weight diagnostics cannot introduce
a NumPy runtime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile

EXTRACTOR_VERSION = "zhao_cui_mlx_extractor_v1"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_P = f"{{{_W_NS}}}p"
_W_PSTYLE = f"{{{_W_NS}}}pStyle"
_W_T = f"{{{_W_NS}}}t"
_W_BR = f"{{{_W_NS}}}br"
_W_TAB = f"{{{_W_NS}}}tab"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == _W_T:
            parts.append(node.text or "")
        elif node.tag == _W_BR:
            parts.append("\n")
        elif node.tag == _W_TAB:
            parts.append("\t")
    return "".join(parts)


def extract_mlx_code(path: Path) -> str:
    """Extract code paragraphs from a MATLAB Live Script without changing text."""
    with zipfile.ZipFile(path) as archive:
        try:
            document = archive.read("matlab/document.xml")
        except KeyError as exc:
            raise ValueError(f"{path} has no matlab/document.xml") from exc
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise ValueError(f"{path} has invalid document.xml") from exc

    blocks: list[str] = []
    for paragraph in root.iter(_W_P):
        style = paragraph.find(f".//{_W_PSTYLE}")
        if style is None or style.attrib.get(f"{{{_W_NS}}}val") != "code":
            continue
        text = _paragraph_text(paragraph)
        if text.strip():
            blocks.append(text)
    code = "\n\n".join(blocks)
    if not code.strip():
        raise ValueError(f"{path} contains no nonempty code paragraph")
    if not re.search(r"\bfunction\b", code):
        raise ValueError(f"{path} code has no MATLAB function declaration")
    return code if code.endswith("\n") else code + "\n"


def extract_mlx_tree(
    source_root: Path,
    output_root: Path,
    model_names: Iterable[str] = ("pp", "sir_austria"),
) -> list[dict[str, Any]]:
    """Extract selected source model Live Scripts and write a hashed manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for model_name in model_names:
        source_dir = source_root / "models" / model_name
        target_dir = output_root / "models" / model_name
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        inputs = sorted(source_dir.glob("*.mlx"))
        if not inputs:
            raise ValueError(f"{source_dir} contains no .mlx callbacks")
        for source_path in inputs:
            code = extract_mlx_code(source_path)
            target_path = target_dir / f"{source_path.stem}.m"
            target_path.write_text(code, encoding="utf-8", newline="")
            records.append(
                {
                    "model": model_name,
                    "source": str(source_path),
                    "derived": str(target_path),
                    "source_sha256": sha256_file(source_path),
                    "derived_sha256": sha256_file(target_path),
                    "bytes": len(code.encode("utf-8")),
                    "classification": "source_faithful_representation_conversion",
                }
            )
    return records


@dataclass(frozen=True)
class LogWeightSummary:
    """Stable scalar diagnostics for a finite importance-weight vector."""

    corrected_logmeanexp: float | None
    legacy_mean_log_weight: float | None
    finite_fraction: float
    importance_ess: float
    valid: bool
    status: str


def summarize_log_weights(values: Iterable[float]) -> LogWeightSummary:
    vals = [float(value) for value in values]
    if not vals:
        return LogWeightSummary(None, None, 0.0, 0.0, False, "empty")
    finite = [value for value in vals if math.isfinite(value)]
    fraction = len(finite) / len(vals)
    legacy = sum(finite) / len(finite) if finite else None
    if len(finite) != len(vals):
        return LogWeightSummary(None, legacy, fraction, 0.0, False, "nonfinite_log_weight")
    maximum = max(vals)
    shifted = [math.exp(value - maximum) for value in vals]
    total = sum(shifted)
    if not math.isfinite(total) or total <= 0:
        return LogWeightSummary(None, legacy, fraction, 0.0, False, "invalid_weight_sum")
    corrected = maximum + math.log(total) - math.log(len(vals))
    normalised = [value / total for value in shifted]
    ess_denominator = sum(value * value for value in normalised)
    ess = 1.0 / ess_denominator if ess_denominator > 0 else 0.0
    valid = math.isfinite(corrected) and math.isfinite(ess)
    return LogWeightSummary(
        corrected if valid else None,
        legacy,
        fraction,
        ess if valid else 0.0,
        valid,
        "ok" if valid else "nonfinite_summary",
    )


def alignment_check(
    source_contract: dict[str, Any],
    current_contract: dict[str, Any],
    required_fields: Iterable[str],
) -> dict[str, Any]:
    """Compare contracts without accepting caller-supplied target labels."""
    fields = list(required_fields)
    mismatches: dict[str, dict[str, Any]] = {}
    for field in fields:
        expected = source_contract.get(field)
        observed = current_contract.get(field)
        if expected is None or observed is None or expected != observed:
            mismatches[field] = {"source": expected, "current": observed}
    status = "aligned" if not mismatches else "not_aligned"
    return {
        "status": status,
        "required_fields": fields,
        "mismatches": mismatches,
        "source_contract": source_contract,
        "current_contract": current_contract,
    }


def derive_full_sol_reference(source_path: Path, target_path: Path, *, stable_logs: bool = True) -> dict[str, Any]:
    """Write a derived class with corrected, fail-closed smooth diagnostics."""
    source_text = source_path.read_text(encoding="utf-8")
    derived = source_text.replace(
        "classdef full_sol < Y_sol",
        "classdef full_sol_reference < Y_sol",
        1,
    ).replace(
        "function sol = full_sol(model, sqr, poly, opt, lowopt, N, epd)",
        "function sol = full_sol_reference(model, sqr, poly, opt, lowopt, N, epd)",
        1,
    )
    old = """            % compute ESS
            w = logpdf_t - logpdf_e;
            % lml = log(mean(exp(w)));
            w_temp = w;
            w_temp(isnan(w_temp)) = [];
            w_temp(isinf(w_temp)) = [];
            lml = mean(w_temp);
            w = exp(w - max(w));
            w = w/sum(w);
            if nargout >= 4
                varargout{1} = logpdf_eall;
            end
            if nargout == 5
                varargout{2} = lml;
            end"""
    new = """            % Reference repair: retain the legacy statistic but compute
            % the finite-sample log marginal likelihood as logmeanexp.
            raw_log_weight = logpdf_t - logpdf_e;
            finite_mask = ~(isnan(raw_log_weight) | isinf(raw_log_weight));
            w_temp = raw_log_weight;
            w_temp(~finite_mask) = [];
            if isempty(w_temp)
                legacy_mean_log_weight = NaN;
            else
                legacy_mean_log_weight = mean(w_temp);
            end
            finite_fraction = sum(finite_mask(:)) / numel(finite_mask);
            if any(~finite_mask(:))
                corrected_status = 0;
                lml = NaN;
                w = zeros(size(raw_log_weight));
                importance_ess = 0;
            else
                shift = max(raw_log_weight);
                shifted = exp(raw_log_weight - shift);
                normalizer = sum(shifted);
                lml = shift + log(normalizer) - log(numel(raw_log_weight));
                w = shifted / normalizer;
                importance_ess = 1 / sum(w.^2);
                corrected_status = isfinite(lml) && isfinite(importance_ess);
                if ~corrected_status
                    w = zeros(size(raw_log_weight));
                    importance_ess = 0;
                end
            end
            if nargout >= 4
                varargout{1} = logpdf_eall;
            end
            if nargout >= 5
                varargout{2} = lml;
            end
            if nargout >= 6
                varargout{3} = struct( ...
                    'raw_log_weight', raw_log_weight, ...
                    'legacy_mean_log_weight', legacy_mean_log_weight, ...
                    'finite_fraction', finite_fraction, ...
                    'importance_ess', importance_ess, ...
                    'corrected_status', corrected_status);
            end"""
    if old not in derived:
        raise ValueError("full_sol smooth block did not match the pinned source")
    derived = derived.replace(old, new, 1)
    if "classdef full_sol_reference < Y_sol" not in derived:
        raise ValueError("derived class declaration missing")
    if "function sol = full_sol_reference" not in derived:
        raise ValueError("derived constructor missing")
    if stable_logs:
        replacements = {
            "logpdf_t = log(priorpdf(sol.model, fulldata(1:d+m, :, 1)));":
                "logpdf_t = reference_logprior(sol.model, fulldata(1:d+m, :, 1));",
            "log(transition(sol.model, fulldata(:, :, k+1), k))":
                "reference_logtransition(sol.model, fulldata(:, :, k+1), k)",
            "log(like(sol.model, fulldata(:, :, k+1), k))":
                "reference_loglike(sol.model, fulldata(:, :, k+1), k)",
        }
        for old_call, new_call in replacements.items():
            if derived.count(old_call) != 1:
                raise ValueError(f"source log-density call changed: {old_call}")
            derived = derived.replace(old_call, new_call, 1)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(derived, encoding="utf-8", newline="")
    return {
        "source": str(source_path),
        "derived": str(target_path),
        "source_sha256": sha256_file(source_path),
        "derived_sha256": sha256_file(target_path),
        "classification": "extension_or_invention_corrected_logmeanexp_diagnostic",
    }


def write_octave_shims(output_root: Path) -> dict[str, Any]:
    """Write the missing Gaussian sampler for the exact source call signature.

    For upper Cholesky R, randn(N,d)*R has covariance R'*R=Sigma.
    This changes RNG consumption versus MATLAB, so numerical seeds are
    reproducible within Octave only, not bitwise across MATLAB and Octave.
    """
    code = """function X = mvnrnd(mu, Sigma, N)
mu = mu(:)';
d = numel(mu);
if ~isequal(size(Sigma), [d, d]) || any(~isfinite(Sigma(:)))
    error('reference mvnrnd: invalid covariance');
end
[R, status] = chol(Sigma);
if status ~= 0, error('reference mvnrnd: covariance must be positive definite'); end
X = bsxfun(@plus, randn(N, d)*R, mu);
end
"""
    path = output_root / "models" / "mvnrnd.m"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return dict(path=str(path), sha256=sha256_file(path),
                classification="extension_or_invention_octave_compatibility",
                mathematical_identity="Cov(Z R) = R' R = Sigma")




def derive_octave_class_overrides(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    """Remove unsupported property annotations, preserving methods and values."""
    import shutil
    source_base = source_root / "deep-tensor.dev/src"
    replacements = {
        "SparseData.m": {"I MultiIndices": "I"},
        "@TTFun/TTFun.m": {"data TTData": "data", "base ApproxBases": "base", "opt  TTOption": "opt"},
        "Polynomials/AdaptLagrangeP.m": {"local LagrangeRef": "local"},
        "Polynomials/Chebyshev2ndUnweighted.m": {"n(1,:)": "n"},
        "Polynomials/Chebyshev1st.m": {"n(1,:)": "n"},
        "Polynomials/LagrangepCDF.m": {"cheby Chebyshev2ndUnweighted": "cheby", "cdf_basis2node(:,:)": "cdf_basis2node"},
    }
    records = []
    for relative, edits in replacements.items():
        source = source_base / relative
        path = output_root / "octave_overrides" / relative
        if source.parent.name.startswith("@"):
            shutil.copytree(source.parent, path.parent)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        code = source.read_text()
        for before, after in edits.items():
            if code.count(before) != 1:
                raise ValueError(f"unexpected property declaration: {relative}: {before}")
            code = code.replace(before, after, 1)
        path.write_text(code)
        records.append(dict(source=str(source), source_sha256=sha256_file(source),
                            derived=str(path), derived_sha256=sha256_file(path),
                            edits=edits, classification="extension_or_invention_octave_property_compatibility"))
    return records


def write_reference_logdensities(output_root: Path) -> list[dict[str, Any]]:
    """Stable closed-form versions of the inspected source Gaussian densities.

    This affects smoothing-weight evaluation only, not TT fitting. Each source
    callback is retained for a cheap equivalence probe where its PDF is positive.
    """
    functions = {
        "reference_logprior.m": """function value = reference_logprior(model, thetax)
if strcmp(model.name, 'pp'), mu = model.pre.init; else, mu = model.pre.priormean; end
residual = bsxfun(@minus, thetax(model.d+1:model.d+model.m,:), mu);
value = -0.5*(model.m*log(2*pi) + sum(residual.^2,1));
if model.d > 0
    value = value - 0.5*(model.d*log(2*pi) + sum(thetax(1:model.d,:).^2,1));
end
end
""",
        "reference_logtransition.m": """function value = reference_logtransition(model, thetax, t)
previous = thetax(model.d+model.m+1:model.d+2*model.m,:);
if strcmp(model.name, 'pp')
    if model.d == 0
        theta = repmat(model.pre.theta,1,size(thetax,2));
    else
        theta = bsxfun(@plus,model.pre.ncons,normcdf(thetax(1:model.d,:)));
    end
    mu = predator_step(model,previous,theta,'RK4');
else
    mu = sir_step(previous,model.pre.theta);
end
residual = (thetax(model.d+1:model.d+model.m,:)-mu)/model.pre.sigma1;
value = -0.5*(model.m*log(2*pi) + 2*model.m*log(model.pre.sigma1) + sum(residual.^2,1));
end
""",
        "reference_loglike.m": """function value = reference_loglike(model, thetax, t)
mu = model.pre.C*thetax(model.d+1:model.d+model.m,:);
residual = bsxfun(@minus,model.Y(:,t),mu)/model.pre.sigma2;
value = -0.5*(model.n*log(2*pi) + 2*model.n*log(model.pre.sigma2) + sum(residual.^2,1));
end
""",
    }
    records = []
    for name, code in functions.items():
        path = output_root / "models" / name
        path.write_text(code)
        records.append(dict(path=str(path), sha256=sha256_file(path),
                            classification="extension_or_invention_stable_gaussian_logdensity"))
    return records


def derive_fixed_target_callbacks(output_root: Path) -> list[dict[str, Any]]:
    """Diagnostic PP conditioning adapter; preserves the author solver classes.

    The source ODE chart is (r,s,u,v,(K-90)/20,(a-20)/10). Removing parameter
    integration and using standard RK4 are explicit changes of the source target.
    """
    model_dir = output_root / "models" / "pp"
    overrides = {
        "priorpdf.m": """function pdf = priorpdf(model, x)
pdf = mvnpdf(x', model.pre.init', eye(model.m))';
end
""",
        "transition.m": """function pdf = transition(model, x, t)
pdf = exp(reference_logtransition(model, x, t));
end
""",
        "st_process.m": """function x = st_process(model, previous, t)
theta = repmat(model.pre.theta, 1, size(previous,2));
x = predator_step(model,previous,theta,'RK4') + model.pre.sigma1*randn(size(previous));
end
""",
    }
    step = model_dir / "predator_step.m"
    original = step.read_text()
    before = "fp4 = reshape(odefun(0, x + fp3*delta/2, theta), 2, []);"
    if original.count(before) != 1:
        raise ValueError("PP source RK fourth stage did not match")
    overrides["predator_step.m"] = original.replace(before, before.replace("delta/2", "delta"))
    records = []
    for name, code in overrides.items():
        path = model_dir / name
        old_hash = sha256_file(path)
        path.write_text(code)
        records.append(dict(path=str(path), before_sha256=old_hash,
                            after_sha256=sha256_file(path),
                            classification="extension_or_invention_fixed_target"))
    return records


def tree_fingerprint(source: Path) -> str:
    """Hash file names and contents to detect mutations anywhere in the snapshot."""
    digest = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
