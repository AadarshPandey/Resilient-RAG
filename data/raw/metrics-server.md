# Metrics Server
The Metrics Server is a scalable, cluster-wide aggregator of resource usage data.
It collects CPU and memory metrics from kubelet Summary APIs across all worker nodes and exposes them via the Metrics API for use by Horizontal Pod Autoscaler and kubectl top commands.
