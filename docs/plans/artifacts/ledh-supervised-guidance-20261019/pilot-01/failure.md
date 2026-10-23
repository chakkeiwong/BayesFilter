# Launch 01: initialization-order repair

No scientific rows were produced. Imported model modules initialized TensorFlow
before the runner could verify memory growth. The fail-closed guard stopped the
process; it was not bypassed. Move scientific imports after the verified memory
policy. Retry in a new output root with unchanged law, controls and data.

Complete stderr is preserved in launch.log. An exact external timer was absent
for this failed launch; charge a conservative 60 seconds, not a fabricated
measured duration. The next launch uses /usr/bin/time. This is an infrastructure
repair trigger, not evidence against supervised guidance.
