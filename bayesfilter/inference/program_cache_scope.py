"""Configuration-time ownership of a compiled endpoint's dependency programs.

Scopes apply during factory construction and tracing, never numerical execution.
Standalone factories retain their bounded compatibility caches. Python ownership
does not imply eviction of TensorFlow's native compiled executables.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

_CURRENT_SCOPE = ContextVar('bayesfilter_program_cache_scope', default=None)


@contextmanager
def independent_trace_scope():
    """Trace a capture-free primitive without retaining the caller FuncGraph.

    TensorFlow init_scope enables eager construction but leaves the default
    graph stack intact. The custom-gradient registry can then retain that
    graph through outer_graph even though the primitive has no tensor captures.
    A fresh graph plus eager context keeps both ancestry and the function-cache
    context independent. This scope is only for shape-only, resource-free code.
    """
    import tensorflow as tf
    from tensorflow.python.eager import context

    with tf.Graph().as_default(), context.eager_mode():
        yield


class ProgramCacheScope:
    """Own the finite set of dependencies built for one endpoint signature."""

    def __init__(self):
        self._programs = {}

    @contextmanager
    def activate(self):
        token = _CURRENT_SCOPE.set(self)
        try:
            yield self
        finally:
            _CURRENT_SCOPE.reset(token)

    def program(self, factory, args, kwargs):
        # The existing lru_cache signatures already require hashable static
        # configuration. Retain keyword ordering just as functools does.
        key = (factory, args, tuple(kwargs.items()))
        if key not in self._programs:
            self._programs[key] = factory(*args, **kwargs)
        return self._programs[key]

    @property
    def program_count(self):
        return len(self._programs)


def scoped_program_cache(*, maxsize):
    """Use the active owner's memo, or the original standalone bounded LRU."""
    def decorate(factory):
        standalone = lru_cache(maxsize=maxsize)(factory)

        @wraps(factory)
        def cached(*args, **kwargs):
            scope = _CURRENT_SCOPE.get()
            if scope is None:
                return standalone(*args, **kwargs)
            return scope.program(factory, args, kwargs)

        cached.cache_info = standalone.cache_info
        cached.cache_clear = standalone.cache_clear
        cached.cache_parameters = standalone.cache_parameters
        return cached

    return decorate
