# Services
A Kubernetes Service is an abstract way to expose an application running on a set of Pods as a network service.
Because pods are ephemeral with dynamic IP addresses, a Service provides a stable IP address and DNS name.
Service types include ClusterIP for internal cluster communication, NodePort to expose ports on each node, LoadBalancer to provision cloud provider balancers, and ExternalName to map to external DNS names.
