"""Independent source checkpoint-transition reference; CPU diagnostic only."""
import copy
from pathlib import Path

def export_transfers(ref, pre_cases, baseline, architecture, common, checkpoint, out):
    torch, yaml = ref.torch, ref.yaml
    # Import only within the diagnostic exporter to keep runtime dependencies separate.
    from reference_kdm_source_stages import build_model
    transitions = []
    specs = {}
    for variant, directory in (("mdpf","experiment0002_importance"),("am_dpf","experiment0003_importance_init")):
        cfg = yaml.safe_load(ref.verify(ref.EXPERIMENT/(directory+"/config.yaml")))
        arch = copy.deepcopy(architecture); arch.update(cfg["models"])
        specs[variant] = arch
    for pre in pre_cases:
        dtype = getattr(torch, pre["dtype"])
        source_params = copy.deepcopy(pre["parameters"])
        source_params.update(pre["updates"][-1]["parameters_after"])
        pre_model = build_model(common["model_params"], specs["mdpf"], checkpoint, dtype)
        with torch.no_grad():
            for name, param in pre_model.named_parameters():
                if name in source_params:
                    param.copy_(torch.tensor(source_params[name],dtype=dtype))
        folder = out/("transfer-"+pre["dtype"]); folder.mkdir()
        components = {}
        for config_key, attribute in (("weight_model","weight_model"),
                ("observation_model","observation_encoder"),
                ("particle_encoder_for_weights_model","particle_encoder_for_weights_model")):
            path = folder/(config_key+".pt")
            torch.save(getattr(pre_model,attribute).state_dict(),path)
            components[config_key] = str(path)
        mdpf = build_model(common["model_params"], specs["mdpf"], checkpoint, dtype)
        mdpf_before = ref.host(dict(mdpf.named_parameters()))
        mdpf.load_pretrained(components, torch.device("cpu"))
        mdpf_after = ref.host(dict(mdpf.named_parameters()))
        full = folder/"mdpf.pt"; torch.save(mdpf.state_dict(),full)
        am = build_model(common["model_params"], specs["am_dpf"], checkpoint, dtype)
        am_before = ref.host(dict(am.named_parameters()))
        am.load_pretrained({"dpf_model":str(full)}, torch.device("cpu"))
        fixtures = {}
        for variant, params in (("mdpf",mdpf_before),("am_dpf",am_before)):
            fixture = copy.deepcopy(next(c for c in baseline["cases"]
                if c["variant"]==variant and c["dtype"]==pre["dtype"]))
            fixture["parameters"] = params
            fixture.pop("updates",None)
            fixtures[variant] = fixture
        transitions.append(dict(dtype=pre["dtype"],pretraining_parameters=source_params,
            pretraining_names=pre["trainable_names"],fixtures=fixtures,
            mdpf_after=mdpf_after,am_dpf_after=ref.host(dict(am.named_parameters()))))
    return transitions
