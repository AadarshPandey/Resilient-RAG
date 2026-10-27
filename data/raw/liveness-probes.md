# Liveness Probes
A liveness probe is used by the kubelet to determine when to restart a container.
If a container is deadlocked or running into an unrecoverable crash loop while remaining alive, the liveness probe fails, and the kubelet terminates the container and restarts it according to its restartPolicy.
