# Cluster Autoscaler
The Cluster Autoscaler automatically adjusts the size of a Kubernetes cluster when there are pods that fail to run due to resource shortages on existing nodes.
It adds worker nodes from cloud node pools when pods are pending, and removes nodes when they are underutilized and their pods can be rescheduled elsewhere.
