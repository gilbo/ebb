---
layout: page
title: "Publications"
date: 
modified:
excerpt:
tags: []
image:
  feature:
---

[**Ebb: A DSL for Physical Simulation on CPUs and GPUs**](http://arxiv.org/abs/1506.07577)
_Gilbert Louis Bernstein, Chinmayee Shah, Crystal Lemire, Zachary DeVito, Matthew Fisher, Philip Levis, Pat Hanrahan_
(under submission)

Designing programming environments for physical simulation is challenging because simulations rely on diverse algorithms and geometric domains. These challenges are compounded when we try to run efficiently on heterogeneous parallel architectures. We present Ebb, a domain-specific language (DSL) for simulation, that runs efficiently on both CPUs and GPUs. Unlike previous DSLs, Ebb uses a three-layer architecture to separate (1) simulation code, (2) definition of data structures for geometric domains, and (3) runtimes supporting parallel architectures. Different geometric domains are implemented as libraries that use a common, unified, relational data model. By structuring the simulation framework in this way, programmers implementing simulations can focus on the physics and algorithms for each simulation without worrying about their implementation on parallel computers. Because the geometric domain libraries are all implemented using a common runtime based on relations, new geometric domains can be added as needed, without specifying the details of memory management, mapping to different parallel architectures, or having to expand the runtime's interface.

We evaluate Ebb by comparing it to several widely used simulations, demonstrating comparable performance to hand-written GPU code where available, and surpassing existing CPU performance optimizations by up to 9x when no GPU code exists.