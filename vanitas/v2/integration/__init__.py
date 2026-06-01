"""Integration shims that inject Vanitas's novel components into third-party
training frameworks without forking them.

Currently: ``slam_mamba_patch`` monkeypatches X-LANCE/SLAM-LLM's ``setup_llm``
so the loaded Qwen3 backbone receives our validated Mamba-2 surgery before
training, keeping SLAM-LLM itself unmodified (clean + reproducible).
"""
