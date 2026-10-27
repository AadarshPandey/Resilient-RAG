# Kube-API-Server
The kube-apiserver is the central control plane component that exposes the Kubernetes HTTP REST API.
It serves as the front end for the control plane, handling authentication, authorization via RBAC, and admission control validation and mutation.
The apiserver is stateless and scales horizontally behind an external load balancer.
It is the only component in the cluster that communicates directly with the etcd storage backend, validating and persisting all resource objects.
