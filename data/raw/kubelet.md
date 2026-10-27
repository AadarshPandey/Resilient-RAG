# Kubelet
The kubelet is the primary node agent running on each worker node in a Kubernetes cluster.
It registers the node with the API server using one of the node's hostnames or network addresses.
The kubelet works in terms of PodSpecs: it receives pod manifests from the apiserver and ensures the containers described in those PodSpecs are running and healthy.
It executes container liveness, readiness, and startup probes, monitors resource usage, and reports pod and node status back to the control plane.
