Restricted Distance calculation
===============================

For a valid partitioning we want to calculate the distances and predecessors between 
all nodes while respecting the restrictions of the partitioning. The restrictions 
are that on a path it is only allowed once to leave and enter a partition.

To recap: We have a graph :math:`G = (V, E)` and a partitioning :math:`P` of
:math:`V` into.

Markdown math: :math:`P = \{P_1, \dots, P_k\}`

The algorithm is based on the paper [1] and the implementation is based on the
implementation of the paper [2].

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```


```{mermaid}
sequenceDiagram
  participant Alice
  participant Bob
  Alice->John: Hello John, how are you?
  loop Healthcheck
      John->John: Fight against hypochondria
  end
  Note right of John: Rational thoughts <br/>prevail...
  John-->Alice: Great!
  John->Bob: How about you?
  Bob-->John: Jolly good!
```