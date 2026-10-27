# DaemonSets
A DaemonSet ensures that all (or some) nodes run a copy of a specific Pod.
As nodes are added to the cluster, Pods are automatically added to them; as nodes are removed, those Pods are garbage collected.
Typical use cases for DaemonSets include running cluster storage daemons like Ceph, log collection daemons like Fluentd, and node monitoring agents like Prometheus node-exporter.
