# Readiness Probes
A readiness probe is used by the kubelet to know when a container is ready to start accepting network traffic.
If a pod's readiness probe fails, the endpoints controller removes the pod's IP address from the endpoints of all matching Services, preventing user requests from reaching unready pods.
