# Container Storage Interface (CSI)
The Container Storage Interface (CSI) is an industry-standard interface that enables third-party storage providers to develop storage plugins for Kubernetes without modifying core repository code.
CSI plugins handle provisioning, attaching, mounting, and snapshotting block and file storage volumes.
Volume attachment is coordinated by the external attacher and provisioner sidecars communicating with the CSI driver running on worker nodes.
