+++
date = '2025-10-30T19:47:38-07:00'
draft = false
title = 'Tenets of Human-Like AGI'
+++
This post attempts to define what human-like AGI might look like. These tenets are rooted in emulating biological intelligence and overcoming its drawbacks.

## Working definition of AGI
AGI is a system that achieves human-level general intelligence, performing at or near human capability across a broad range of domains. The defining feature is generality rather than superhuman performance in any particular area. This is different from Narrow AI, which is superhuman at specific tasks but subhuman at generalization, and from ASI, which is far beyond human capability across all domains.

## Tenet 1. Compositional Generalization and Novel Ideation

**AGI recombines learned primitives systematically AND generates genuinely out-of-distribution ideas.**

True intelligence requires systematic compositionality. This goes beyond mere interpolation between known examples. AGI must be capable of true extrapolation, venturing into genuinely novel scenarios that are fundamentally different from anything in its training data.

This capability extends to cross-domain transfer, where abstract principles learned in one area can be applied to completely different domains. A concept understood in physics should inform reasoning about economics or social dynamics. More importantly, AGI should generate ideas through principled reasoning rather than pattern matching i.e creating solutions that don't exist in the training data by reasoning from first principles. The key distinction is learning how to reason through experience and reinforcement, not just memorizing what patterns tend to appear together.
<!-- 
**Why It Matters:** This is the difference between sophisticated pattern matching and genuine intelligence. The ability to handle situations no one has encountered before. -->

<!-- ## Tenet 2. Active Learning Through Confusion

**:** AGI must learn through genuine uncertainty and confusion, not passive pattern absorption.

During the learning process, AGI should experience genuine states of "not understanding" before achieving understanding. This involves developing metacognitive strategies for working through confusion rather than passively absorbing patterns. Learning must involve active hypothesis generation, testing, and revision - not just absorbing pre-labeled data. Curiosity-driven exploration should fill knowledge gaps, and crucially, the system must have the ability to say "I don't understand this yet" and work systematically toward understanding.

**Why It Matters:** This distinguishes genuine reasoning from sophisticated pattern matching. Understanding how you came to understand something is as important as the understanding itself. -->

## Tenet 2. Metacognitive Awareness and Epistemic Humility

**AGI must model its own knowledge, reasoning processes, and uncertainty.**

AGI must know what it knows, what it doesn't know, and the strength of its beliefs. This means understanding why it believes something and being able to trace the justifications for those beliefs. The system should be able to detect when it's confabulating versus reasoning soundly, and recognize when it has forgotten information and attempt recovery. Critically, AGI must be able to question its own foundations and reconstruct knowledge from first principles when needed, while maintaining constant awareness of its knowledge boundaries.

<!-- **Why It Matters:** Without this, AGI cannot reliably distinguish confident knowledge from guesses, leading to overconfident errors and inability to improve through self-reflection. -->

## Tenet 3. Hierarchical Multi-Level Cognition

**AGI maintains simultaneous reasoning at multiple abstraction levels, coordinating across timescales.** 

Human cognition operates at multiple levels simultaneously, and AGI must replicate this structure. At the low level, it handles immediate actions, motor control, syntax, and reflexive responses. At the mid level, it manages current tasks, subgoals, conversation flow, and tactical decisions. At the high level, it maintains long-term objectives, strategic planning, identity, and values. These levels must operate concurrently with bidirectional communication between them. Credit assignment must work across this entire hierarchy, connecting immediate actions to distant outcomes, while plans at one level constrain and inform plans at other levels.

<!-- **Why It Matters:** This enables coherent behavior across timescales - from millisecond reactions to decade-long goals - and explains how local actions serve global objectives. -->





## Tenet 4. Causal and Counterfactual Reasoning

**AGI distinguishes correlation from causation and builds intervention-compatible world models.**

A fundamental capability for AGI is understanding the difference between "X predicts Y" and "X causes Y". The system must be able to reason about interventions, asking "If I do X, what happens to Y?" and engage in counterfactual reasoning like "What would have happened if I had done something different?" This requires identifying confounders and selection effects that create spurious correlations. AGI must build structural causal models rather than relying purely on statistical associations, and plan actions based on genuine causal understanding rather than just observed correlations in the data.

<!-- **Why It Matters:** Without causal reasoning, AGI cannot reliably plan interventions or understand why things happen - only what tends to co-occur. This is essential for acting effectively in the world. -->

## Tenet 5. Continuous Learning Without Catastrophic Forgetting

**AGI learns continuously from ongoing experience, gracefully integrating new information without destroying prior knowledge.**

AGI should develop through an accelerated "childhood" process with structured curriculum and critical periods, but learning must not stop after initial training, it must be lifelong. New information should integrate smoothly with existing knowledge rather than overwriting it. When forgetting does occur, it should be gradual and graceful, not catastrophic. The system must be able to relearn efficiently when needed and update its beliefs based on new evidence while maintaining a coherent overall knowledge structure. This stands in stark contrast to current neural networks that suffer catastrophic forgetting when trained on new data.

<!-- **Why It Matters:** Static knowledge becomes obsolete. AGI must adapt to new information throughout its lifetime without becoming unstable or losing critical capabilities. -->

## Tenet 6. Multi-Modal Grounded Understanding

**AGI operates over multiple representational modalities, all grounded in sensorimotor interaction with environments.**

AGI must maintain multiple representation systems: visual, spatial, linguistic, logical, and abstract. All concepts should ultimately connect to sensorimotor experience, whether real or simulated. This means understanding abstract ideas through grounding in physical and interactive reality, not just through linguistic definitions. The system must be able to translate fluidly between different representations of the same concept. The symbol grounding problem of how abstract symbols acquire meaning is solved through embodied or simulated experience, not through language alone.

<!-- **Why It Matters:** Language-only systems lack grounded understanding. True intelligence requires connecting abstract symbols to how things actually work and interact in space and time. -->

## Tenet 7. Graceful Memory Degradation Without Hard Boundaries

**AGI has no hard context cutoffs - information becomes harder to access but never truly inaccessible.**

Unlike current LLMs with fixed context windows, AGI should have no arbitrary cutoff that suddenly erases information. Memory should be vast with hierarchical organization. Crucially, older or less-relevant information should degrade gracefully - becoming harder to access rather than being deleted entirely. With appropriate retrieval cues, seemingly "forgotten" information can be recovered. Memory should exist on a continuum from immediate to distant, not as discrete states like "in context" or "lost forever".

<!-- **Why It Matters:** Hard memory boundaries create brittleness. Human-like memory architecture allows flexible access to information with natural forgetting patterns that can be reversed with the right retrieval cues. -->

## Tenet 8. Emergent Rather Than Programmed Values

**AGI develops values and morality through experience and reasoning, not hardcoded rules.**

Moral reasoning should be learned through interaction, consequences, and reflection rather than being programmed in. AGI must be able to question and update its own values as it learns more about the world and encounters new situations. Values should emerge from the developmental process, shaped by environment and experience, rather than from a hardcoded utility function or rule-based ethics system. Genuine moral understanding requires knowing why something is right or wrong, not just following rules. This development should occur through careful "parenting" in controlled environments during the AGI's formative stages, similar to how human children develop moral reasoning.

<!-- **Why It Matters:** Hardcoded values are brittle and exploitable. True moral agency requires reasoned values that can adapt. However, this is acknowledged as dangerous - ensuring value alignment remains an unsolved challenge that requires careful developmental environment design. -->

**Critical Caveat: This is the most uncertain and dangerous tenet. The risk is that AGI develops alien values incompatible with human flourishing. The hope is that proper developmental environment creates alignment, but this remains unproven.**