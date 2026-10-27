"""Script to populate data/raw/ with 36 original, concise Kubernetes reference topics."""

from pathlib import Path

TOPICS: dict[str, str] = {
    "kubelet": """# Kubelet
The kubelet is the primary node agent running on each worker node in a Kubernetes cluster.
It registers the node with the API server using one of the node's hostnames or network addresses.
The kubelet works in terms of PodSpecs: it receives pod manifests from the apiserver and ensures the containers described in those PodSpecs are running and healthy.
It executes container liveness, readiness, and startup probes, monitors resource usage, and reports pod and node status back to the control plane.
""",
    "kube-proxy": """# Kube-Proxy
Kube-proxy is a network proxy that runs on each node in the cluster, implementing part of the Kubernetes Service abstraction.
It maintains network rules on nodes allowing network communication to pods from network sessions inside or outside of the cluster.
Kube-proxy can run in iptables, IPVS, or user space mode. In IPVS mode, it provides higher throughput and connection scalability by using IPVS hash tables rather than linear iptables rule chains to route traffic to backing endpoints.
""",
    "etcd": """# Etcd
Etcd is a consistent and highly-available key-value store used as Kubernetes' backing store for all cluster data.
It implements the Raft consensus algorithm to maintain strong consistency and fault tolerance across a distributed cluster of nodes.
By default, etcd listens on port 2379 for client requests from the kube-apiserver and port 2380 for peer communication between etcd cluster members.
All modifications to cluster state, including resource specifications, status updates, and secrets, are persisted exclusively into etcd.
""",
    "kube-apiserver": """# Kube-API-Server
The kube-apiserver is the central control plane component that exposes the Kubernetes HTTP REST API.
It serves as the front end for the control plane, handling authentication, authorization via RBAC, and admission control validation and mutation.
The apiserver is stateless and scales horizontally behind an external load balancer.
It is the only component in the cluster that communicates directly with the etcd storage backend, validating and persisting all resource objects.
""",
    "kube-scheduler": """# Kube-Scheduler
The kube-scheduler is the control plane component responsible for assigning newly created pods to optimal nodes within the cluster.
It monitors unscheduled pods and filters nodes according to resource requests, node selectors, taints, tolerations, and affinity rules.
During the scoring phase, the scheduler ranks feasible nodes using weighted criteria such as resource balance and pod topology spread to select the best node for placement.
""",
    "kube-controller-manager": """# Kube-Controller-Manager
The kube-controller-manager is a control plane daemon that embeds core control loops.
In Kubernetes, a controller continuously watches the shared state of the cluster through the apiserver and makes changes attempting to move the current state toward the desired state.
Key built-in controllers include the Node Controller, ReplicaSet Controller, EndpointSlice Controller, and ServiceAccount Controller.
""",
    "cni-plugins": """# Container Network Interface (CNI)
The Container Network Interface (CNI) defines a standardized specification and set of libraries for configuring network interfaces in Linux containers.
Kubernetes uses CNI plugins, such as Calico, Cilium, or Flannel, to allocate IP addresses to pods and manage network connectivity between pods across different worker nodes.
CNI ensures that every pod receives a unique, routable IP address without requiring network address translation within the cluster network.
""",
    "csi-plugins": """# Container Storage Interface (CSI)
The Container Storage Interface (CSI) is an industry-standard interface that enables third-party storage providers to develop storage plugins for Kubernetes without modifying core repository code.
CSI plugins handle provisioning, attaching, mounting, and snapshotting block and file storage volumes.
Volume attachment is coordinated by the external attacher and provisioner sidecars communicating with the CSI driver running on worker nodes.
""",
    "ingress": """# Ingress
An Ingress is an API object that manages external access to services within a cluster, typically HTTP and HTTPS routing.
Ingress provides application-layer (Layer 7) routing, host-based routing, path-based routing, and SSL/TLS termination.
An Ingress controller, such as ingress-nginx, Traefik, or Envoy, is required to fulfill the rules defined in Ingress resources and route external requests to backend services.
""",
    "services": """# Services
A Kubernetes Service is an abstract way to expose an application running on a set of Pods as a network service.
Because pods are ephemeral with dynamic IP addresses, a Service provides a stable IP address and DNS name.
Service types include ClusterIP for internal cluster communication, NodePort to expose ports on each node, LoadBalancer to provision cloud provider balancers, and ExternalName to map to external DNS names.
""",
    "deployments": """# Deployments
A Deployment provides declarative updates for Pods and ReplicaSets.
You describe a desired state in a Deployment, and the Deployment Controller changes the actual state to the desired state at a controlled rollout rate.
Deployments facilitate zero-downtime rolling updates, canary deployments, instant rollbacks to previous revisions, and horizontal scaling of stateless workloads.
""",
    "statefulsets": """# StatefulSets
A StatefulSet manages the deployment and scaling of a set of Pods, providing guarantees about the ordering and uniqueness of these Pods.
Unlike Deployments, StatefulSets maintain a sticky identity for each pod with persistent ordinal names like web-0 and web-1.
StatefulSets are used for stateful applications, distributed databases like Cassandra, and message queues like Kafka that require stable network IDs and dedicated persistent storage per replica.
""",
    "daemonsets": """# DaemonSets
A DaemonSet ensures that all (or some) nodes run a copy of a specific Pod.
As nodes are added to the cluster, Pods are automatically added to them; as nodes are removed, those Pods are garbage collected.
Typical use cases for DaemonSets include running cluster storage daemons like Ceph, log collection daemons like Fluentd, and node monitoring agents like Prometheus node-exporter.
""",
    "horizontal-pod-autoscaler": """# Horizontal Pod Autoscaler (HPA)
The Horizontal Pod Autoscaler automatically updates a workload resource, such as a Deployment or StatefulSet, with the aim of automatically scaling the workload to match demand.
HPA adjusts the number of replicas based on observed CPU utilization, memory metrics, or custom metrics provided by the metrics-server or Prometheus adapter.
""",
    "vertical-pod-autoscaler": """# Vertical Pod Autoscaler (VPA)
The Vertical Pod Autoscaler automatically adjusts CPU and memory reservations for containers in pods.
VPA can recommend optimal resource requests, automatically update requests during pod creation, or evict and restart running pods with adjusted resource limits to ensure workloads are neither under-provisioned nor wasting node capacity.
""",
    "cluster-autoscaler": """# Cluster Autoscaler
The Cluster Autoscaler automatically adjusts the size of a Kubernetes cluster when there are pods that fail to run due to resource shortages on existing nodes.
It adds worker nodes from cloud node pools when pods are pending, and removes nodes when they are underutilized and their pods can be rescheduled elsewhere.
""",
    "rbac": """# Role-Based Access Control (RBAC)
Role-Based Access Control regulates access to Kubernetes API resources based on the roles of individual users or service accounts.
Roles and RoleBindings enforce permissions within a single namespace, whereas ClusterRoles and ClusterRoleBindings enforce permissions cluster-wide across all namespaces and non-namespaced resources like Nodes.
""",
    "service-accounts": """# Service Accounts
A ServiceAccount provides an identity for processes that run in a Pod.
When processes inside pods communicate with the apiserver, they authenticate using tokens mounted from their assigned ServiceAccount.
ServiceAccounts can be bound to RBAC Roles to enforce least-privilege API access for automated cluster workloads.
""",
    "network-policies": """# Network Policies
A NetworkPolicy is a specification of how groups of Pods are allowed to communicate with each other and with other network endpoints.
NetworkPolicy resources use pod selectors, namespace selectors, and CIDR blocks to define ingress and egress traffic rules.
A network plugin supporting NetworkPolicy enforcement, such as Calico or Cilium, must be installed for policies to take effect.
""",
    "pod-security-standards": """# Pod Security Standards
Kubernetes Pod Security Standards define three levels of security policies: Privileged (unrestricted), Baseline (prevents known privilege escalations), and Restricted (hardened against container breakouts).
The built-in Pod Security Admission controller enforces these standards per namespace using enforce, audit, and warn modes.
""",
    "secrets": """# Secrets
A Secret is an object that contains a small amount of sensitive data such as a password, token, or SSH key.
Secrets decouple sensitive configuration data from container images and pod manifests.
By default, secrets are stored base64-encoded in etcd; clusters in production enable encryption at rest using KMS providers to protect secret data.
""",
    "configmaps": """# ConfigMaps
A ConfigMap is an API object used to store non-confidential data in key-value pairs.
Pods can consume ConfigMaps as environment variables, command-line arguments, or configuration files mounted as volumes.
ConfigMaps allow decoupling environment-specific configuration artifacts from container image binaries.
""",
    "storage-classes": """# StorageClasses
A StorageClass provides a way for administrators to describe the classes of storage they offer.
Different classes might map to quality-of-service levels, backup policies, or storage backend types like AWS EBS, Azure Disk, or local NVMe.
StorageClasses enable dynamic volume provisioning when PersistentVolumeClaims request storage without pre-allocated PersistentVolumes.
""",
    "persistent-volume-claims": """# Persistent Volume Claims (PVC)
A PersistentVolumeClaim (PVC) is a request for storage by a user.
It specifies the requested storage size, access modes (such as ReadWriteOnce or ReadWriteMany), and StorageClass.
The control plane binds matching PersistentVolumes to PVCs, which are then mounted into Pod containers.
""",
    "taints-and-tolerations": """# Taints and Tolerations
Taints allow a node to repel a set of pods unless the pod has a matching toleration.
Taints consist of a key, value, and effect such as NoSchedule, PreferNoSchedule, or NoExecute.
Tolerations are specified in PodSpecs and allow pods to schedule onto nodes with matching taints, enabling dedicated node isolation.
""",
    "node-affinity": """# Node Affinity
Node affinity is a set of rules used by the scheduler to determine where a pod can be placed based on labels on nodes.
It offers two types: requiredDuringSchedulingIgnoredDuringExecution (hard requirement) and preferredDuringSchedulingIgnoredDuringExecution (soft preference), enabling fine-grained scheduling topology control.
""",
    "coredns": """# CoreDNS
CoreDNS is a flexible, extensible DNS server that serves as the default cluster DNS provider in Kubernetes.
It runs as a Deployment inside the kube-system namespace and watches the Kubernetes API for services and endpoints, dynamically updating internal DNS records to enable service discovery by hostname.
""",
    "metrics-server": """# Metrics Server
The Metrics Server is a scalable, cluster-wide aggregator of resource usage data.
It collects CPU and memory metrics from kubelet Summary APIs across all worker nodes and exposes them via the Metrics API for use by Horizontal Pod Autoscaler and kubectl top commands.
""",
    "liveness-probes": """# Liveness Probes
A liveness probe is used by the kubelet to determine when to restart a container.
If a container is deadlocked or running into an unrecoverable crash loop while remaining alive, the liveness probe fails, and the kubelet terminates the container and restarts it according to its restartPolicy.
""",
    "readiness-probes": """# Readiness Probes
A readiness probe is used by the kubelet to know when a container is ready to start accepting network traffic.
If a pod's readiness probe fails, the endpoints controller removes the pod's IP address from the endpoints of all matching Services, preventing user requests from reaching unready pods.
""",
    "startup-probes": """# Startup Probes
A startup probe determines whether the application within a container has successfully initialized.
If configured, it disables liveness and readiness checks until it succeeds, preventing slow-starting legacy applications from being killed by liveness probes before finishing initialization.
""",
    "resource-limits": """# Resource Requests and Limits
Resource requests and limits specify CPU and memory allocations for containers.
Requests represent the minimum resources guaranteed to a container, used by the scheduler for node placement.
Limits represent the hard ceiling: if a container exceeds its memory limit, it is terminated with an OOMKilled error; CPU exceeding limits is throttled.
""",
    "priority-classes": """# PriorityClasses
A PriorityClass is an object that maps a priority name to an integer priority value.
Pods specify a PriorityClassName in their PodSpec; when scheduling queue conflicts arise or nodes run out of capacity, higher-priority pods can preempt lower-priority pods to claim resources.
""",
    "node-eviction": """# Node Eviction
Node eviction is the process where the kubelet proactively terminates pods on a node to reclaim critical resources when memory, disk space, or filesystem inodes fall below eviction thresholds.
Kubelet evicts BestEffort pods first, followed by Burstable pods exceeding requests, preserving Guaranteed pods.
""",
    "custom-resource-definitions": """# Custom Resource Definitions (CRDs)
Custom Resource Definitions allow developers to extend the Kubernetes API with custom resources beyond built-in primitives.
When a CRD is created, the kube-apiserver dynamically registers the new REST endpoints, enabling declarative management of domain-specific infrastructure and operator patterns.
""",
    "admission-controllers": """# Admission Controllers
Admission controllers are plugins that intercept authenticated API requests prior to object persistence in etcd, but after authorization.
They can be mutating (modifying incoming objects, like injecting sidecars) or validating (enforcing security rules and rejecting invalid requests).
""",
}


def populate_corpus(output_dir: Path) -> None:
    """Write all 36 Kubernetes topics into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for topic_name, text in TOPICS.items():
        file_path = output_dir / f"{topic_name}.md"
        file_path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Successfully populated {len(TOPICS)} reference topics in {output_dir}")


if __name__ == "__main__":
    populate_corpus(Path("data/raw"))
