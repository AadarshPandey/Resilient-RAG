# Kube-Controller-Manager
The kube-controller-manager is a control plane daemon that embeds core control loops.
In Kubernetes, a controller continuously watches the shared state of the cluster through the apiserver and makes changes attempting to move the current state toward the desired state.
Key built-in controllers include the Node Controller, ReplicaSet Controller, EndpointSlice Controller, and ServiceAccount Controller.
