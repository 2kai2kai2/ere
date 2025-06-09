## 0.2.2

-   Added u8 one-pass engine.
-   Improved performance via suffix optimizations (merging states with the same outgoing transitions).
-   Fixed a bug where the first branch for priority-shortest 'upto' node NFA generation had priority inverted.
-   Improved performance by actually implementing the greedy-mode thread culling for PikeVM
-   Added this changelog.

## 0.2.1

-   Added unstable struct attribute-based method for defining regexes.

## 0.2.0

-   Added PikeVM engine (including `exec` functionality).
-   Added U8PikeVM engine (including `exec` functionality).

## 0.1.0

-   Added initial parsing.
-   Added initial `test` functionality on basic engine.
