# Kube-Proxy
Kube-proxy is a network proxy that runs on each node in the cluster, implementing part of the Kubernetes Service abstraction.
It maintains network rules on nodes allowing network communication to pods from network sessions inside or outside of the cluster.
Kube-proxy can run in iptables, IPVS, or user space mode. In IPVS mode, it provides higher throughput and connection scalability by using IPVS hash tables rather than linear iptables rule chains to route traffic to backing endpoints.
