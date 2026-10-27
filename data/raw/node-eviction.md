# Node Eviction
Node eviction is the process where the kubelet proactively terminates pods on a node to reclaim critical resources when memory, disk space, or filesystem inodes fall below eviction thresholds.
Kubelet evicts BestEffort pods first, followed by Burstable pods exceeding requests, preserving Guaranteed pods.
