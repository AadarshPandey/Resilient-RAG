# StatefulSets
A StatefulSet manages the deployment and scaling of a set of Pods, providing guarantees about the ordering and uniqueness of these Pods.
Unlike Deployments, StatefulSets maintain a sticky identity for each pod with persistent ordinal names like web-0 and web-1.
StatefulSets are used for stateful applications, distributed databases like Cassandra, and message queues like Kafka that require stable network IDs and dedicated persistent storage per replica.
