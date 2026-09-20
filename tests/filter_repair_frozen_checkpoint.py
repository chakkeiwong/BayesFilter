"""Diagnostic source loader with an isolated, Git-pinned numerical closure.

Only direct numerical-module imports are supported. Package namespaces are
empty containers, so public lazy-export registries cannot resolve current code.
Third-party imports use the same installed environment as the candidate.
"""

import builtins
import hashlib
import subprocess
import sys
import types


class FrozenCheckpoint:
    def __init__(self, revision, label):
        self.revision = revision
        self.prefix = f"filter_repair_{label}_{revision}"
        self.modules = {}
        self.sources = {}

    def load(self, name):
        if name in self.modules:
            return self.modules[name]
        if not name.startswith("bayesfilter"):
            raise ImportError(name)
        alias = self.prefix + "." + name
        module = types.ModuleType(alias)
        self.modules[name] = module
        sys.modules[alias] = module  # Dataclass/type metadata, isolated from runtime.
        if name in ("bayesfilter", "bayesfilter.inference", "bayesfilter.ops"):
            module.__path__ = []
            return module
        path = name.replace(".", "/") + ".py"
        source = subprocess.check_output(["git", "show", f"{self.revision}:{path}"], text=True)
        self.sources[path] = source
        module.__file__ = f"{self.revision}:{path}"
        module.__builtins__ = {**vars(builtins), "__import__": self._import}
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - exact frozen diagnostic code
        return module

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise ImportError("Relative imports require an explicit frozen-source binding")
        if name != "bayesfilter" and not name.startswith("bayesfilter."):
            return builtins.__import__(name, globals, locals, fromlist, level)
        if not fromlist or "*" in fromlist:
            raise ImportError("Only explicit from-imports are admitted by this diagnostic loader")
        module = self.load(name)
        for attribute in fromlist:
            if not hasattr(module, attribute):
                setattr(module, attribute, self.load(name + "." + attribute))
        return module

    def hashes(self):
        return {path: hashlib.sha256(source.encode()).hexdigest()
            for path, source in sorted(self.sources.items())}
