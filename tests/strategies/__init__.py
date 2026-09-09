"""Parliament-specific Hypothesis strategies.

Hypothesis owns generation, shrinking, and replay. Modules here only encode
legal world structure and declared topology. Production validators must not
import this package to decide what a world contains.
"""
