def thinking_process_generation_prompt(problem, concepts, difficulty_level):
    concept_text = "\n".join(f"{i+1}. {concept}" for i, concept in enumerate(concepts))
    prompt = (
        "Imagine you are an expert in educational problem design.\n"
        f"You will be shown these components:\n\n"
        f"Problem: {problem}\n\n"
        f"Foundamental Concepts:\n{concept_text}\n\n"
        f"Difficulty Level: {difficulty_level}\n\n"
        "Your task is to reverse-engineer a clear thinking process that shows how a teacher might design this problem. This thinking process should:\n"
        "- Show how combining the given foundational concepts naturally leads to a problem at the specified difficulty level\n"
        "- Include all key decisions and reasoning that shaped the problem design\n"
        "- (IMPORTANT) The thinking process must be so precise and detailed that another teacher following these exact steps would recreate the identical problem\n"
        "- (IMPORTANT) The thinking process must be so natural and logical that another teacher could derive the same thinking process using only the foundational concepts and difficulty level\n\n"
        "Present your answer after 'Thinking Process: ' with the complete step-by-step thinking process described above."
    )

    return prompt
