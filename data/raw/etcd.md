# Etcd
Etcd is a consistent and highly-available key-value store used as Kubernetes' backing store for all cluster data.
It implements the Raft consensus algorithm to maintain strong consistency and fault tolerance across a distributed cluster of nodes.
By default, etcd listens on port 2379 for client requests from the kube-apiserver and port 2380 for peer communication between etcd cluster members.
All modifications to cluster state, including resource specifications, status updates, and secrets, are persisted exclusively into etcd.
