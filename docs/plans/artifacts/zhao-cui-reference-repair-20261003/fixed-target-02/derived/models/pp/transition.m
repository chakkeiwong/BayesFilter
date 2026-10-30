function pdf = transition(model, x, t)
pdf = exp(reference_logtransition(model, x, t));
end
