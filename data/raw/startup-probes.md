# Startup Probes
A startup probe determines whether the application within a container has successfully initialized.
If configured, it disables liveness and readiness checks until it succeeds, preventing slow-starting legacy applications from being killed by liveness probes before finishing initialization.
