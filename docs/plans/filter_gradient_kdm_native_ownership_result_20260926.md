# KDM native ownership and bounded process lifetime

The enclosing-only mixture experiment does not repair the construction-memory
growth and is not installed in runtime. Runs 03976--03983 attribute the growth
primarily to live glibc heap allocation after complete outputs and Python owners
are released. Six fresh XLA owners continue to retain host resources on both
CPU and GPU 2; the graph controls level off. Each owner produces exact replay,
and registered-function counts remain constant after collection.

| Device/mode | Ownership | RSS after owner 1 / 6, MiB | Late live glibc allocation, MiB per owner |
| --- | --- | ---: | ---: |
| CPU graph | nested | 768 / 840 | 0.01 |
| CPU graph | enclosing only | 764 / 833 | 0.01 |
| CPU XLA | nested | 1210 / 2580 | 251.93 |
| CPU XLA | enclosing only | 1201 / 2545 | 249.18 |
| GPU XLA | nested | 1360 / 2478 | 199.93 |
| GPU XLA | enclosing only | 1353 / 2457 | 197.44 |

Late allocation is the difference between owners 3 and 6 divided by three.
Most extra resident pages are in the heap mapping: 751 MiB CPU / 599 MiB GPU
over the last three nested builds. Device live allocation returns to 7168
bytes. These categories distinguish live native allocation from free glibc
pages and live GPU tensors; they do not name an exact C++ allocation owner.

The installed TensorFlow 2.19.1 header
`tensorflow/compiler/jit/device_compilation_cache.h:62-66` says that the cache
owns compiled HLO, executables and metadata and has no eviction policy.
`device_compiler.h:439-446` describes cache lookup by compilation signature.
The explicit logging worker 03981 reports exactly six completed compilations,
one per owner, despite twelve numerical calls. Detailed cache-entry logging
was unavailable; 03980 also had pytest output capture enabled. Preserve those
limitations. The evidence is consistent with a process-lived compilation cache;
it does not prove which native object owns every retained byte.

The next repair mechanism is an explicit caller lifecycle: keep one owner for
fixed callbacks/configuration, pass varying numerical operands as tensors, and
end the worker process when changing that fixed configuration. 03984 checks
three serial CPU children at callback scales 1, 1.01, 1. Each uses one XLA owner
for 20 exact replays, checks its complete public record against the frozen
original, accepts changed tensor inputs without retracing, and exits cleanly.
The first and third children have identical complete results; the changed
callback has a different result matching its own original reference. Native
allocated bytes remain about 504 MiB per child and do not grow across the 20
calls; the supervisor RSS remains 580--581 MiB after child exits. GPU run 03985
passes the same gates. Child live glibc allocations remain constant within each
20-call run (655--671 MiB), and parent RSS remains 859--861 MiB across exits.
GPU child current allocation stays 20 KiB while the retained outputs are live.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
| --- | --- | --- | --- | --- | --- |
| Reject enclosing-only ownership as the memory repair | Repeated-construction slope persists | Memory repair fails | Small compiler-structure cost differences are unqualified | Retain current runtime | No full candidate/HLO qualification or cost ranking |
| Document explicit bounded worker lifetime | CPU/GPU complete records, replay and reaping pass | No fixture veto | Callback/configuration scope and target scale | Keep one owner per fixed configuration; end workers between configurations | No automatic subprocess wrapper or cache mutation |
| Continue call-chain execution repairs | Outer source guard omits safety/helper modules | Whole-repo compliance remains open | Remaining stage/reference recurrences | Execute linked safety/stage plan | No LEDH admission or main promotion |

Red-team review: terminating a worker bounds process-owned retention but does
not repair TensorFlow cache eviction or make repeated public construction safe
inside a long-lived process. The measurements are one bounded diagnostic cohort,
not a statistical performance ranking. Full original/repaired XLA numerical
qualification remains the earlier KDM result; the precision gate and wider
master findings remain open.

03986 passes all 129 policy checks. The public factory documentation now warns
that executable memory can survive Python collection and describes explicit
owner/process lifetimes. The numerical runtime is unchanged. The 11 supervised
invocations contain 8 CPU and 3 GPU jobs; each lifecycle job includes three
serial numerical children and charges their entire parent elapsed time. Thus
there are 10 CPU and 5 GPU numerical processes, not 11 numerical workers. This
clarifies the earlier plan's ambiguous worker count without expanding its
1800/900-second suballocation or the global 56/52-hour caps.

This unit charged 351.184418 CPU / 207.219570 GPU seconds. All 71 archived files
and executed diagnostic source revisions verify byte-for-byte against the
03986 receipt. No numerical worker is active at this checkpoint.
