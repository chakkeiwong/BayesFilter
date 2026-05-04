from bayesfilter.adapters import dsge_structural_adapter_gate


class MissingDSGEMetadata:
    pass


class ToyDSGEMetadata:
    bayesfilter_state_names = ("m", "k")
    bayesfilter_stochastic_indices = (0,)
    bayesfilter_deterministic_indices = (1,)
    bayesfilter_innovation_dim = 1

    def bayesfilter_deterministic_completion(self, previous_state, stochastic_state):
        return stochastic_state


class SmallNKLikeAllStochastic:
    bayesfilter_state_names = ("a", "v")
    bayesfilter_stochastic_indices = (0, 1)
    bayesfilter_deterministic_indices = ()
    bayesfilter_innovation_dim = 2


class RotembergLikeMixedMissingCompletion:
    bayesfilter_state_names = ("R", "g", "z", "dy")
    bayesfilter_stochastic_indices = (0, 1, 2)
    bayesfilter_deterministic_indices = (3,)
    bayesfilter_innovation_dim = 3


class MixedWithoutCompletion(ToyDSGEMetadata):
    bayesfilter_deterministic_completion = None


def test_dsge_adapter_gate_fails_closed_without_explicit_metadata():
    result = dsge_structural_adapter_gate(MissingDSGEMetadata())

    assert result.adapter_ready is False
    assert result.partition is None
    assert "missing explicit DSGE structural metadata" in result.blockers[0]


def test_dsge_adapter_gate_accepts_explicit_structural_metadata():
    result = dsge_structural_adapter_gate(ToyDSGEMetadata())

    assert result.adapter_ready is True
    assert result.partition is not None
    assert result.partition.state_names == ("m", "k")
    assert result.partition.stochastic_indices == (0,)
    assert result.partition.deterministic_indices == (1,)
    assert result.partition.innovation_dim == 1
    assert result.metadata_regime == "mixed_structural"


def test_dsge_adapter_gate_blocks_mixed_model_without_completion_map():
    result = dsge_structural_adapter_gate(MixedWithoutCompletion())

    assert result.adapter_ready is False
    assert "deterministic completion map" in result.blockers[0]


def test_dsge_adapter_gate_accepts_small_nk_like_all_stochastic_metadata():
    result = dsge_structural_adapter_gate(SmallNKLikeAllStochastic())

    assert result.adapter_ready is True
    assert result.metadata_regime == "all_stochastic"
    assert result.partition.state_names == ("a", "v")
    assert result.partition.stochastic_indices == (0, 1)
    assert result.partition.deterministic_indices == tuple()


def test_dsge_adapter_gate_blocks_rotemberg_like_mixed_metadata_without_completion():
    result = dsge_structural_adapter_gate(RotembergLikeMixedMissingCompletion())

    assert result.adapter_ready is False
    assert result.metadata_regime == "mixed_structural"
    assert "deterministic completion map" in result.blockers[0]
