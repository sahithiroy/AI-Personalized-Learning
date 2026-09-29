"""PLRS - Personalized Learning and Remedial recommendation System.

Implementation of "AI-Driven Personalized Learning and Remedial Recommendation
Through Knowledge Concept-Centric Evaluation" (Pradeesh et al., IEEE Access, 2025).

Modules
-------
rag         Curriculum PDF -> chunks -> embeddings -> vector DB, concept extraction
kt          Exercise-aware Knowledge Tracing (EKT) + DKT baseline
gap         OBE threshold/target based knowledge-gap analysis
qgen        RAG MCQ generation, semantic de-duplication, cross-verification
remedial    Generative remedial recommendations with cross-verification
evaluation  Concept-based adaptive evaluation, metrics and statistics
pipeline    End-to-end orchestration (Algorithm 1)
"""

__version__ = "1.0.0"
