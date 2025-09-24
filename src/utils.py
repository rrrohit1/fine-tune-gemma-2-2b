from transformers import AutoTokenizer

def format_chat(
    messages,
    tokenizer,
    tools=None,
    add_generation_prompt=True,
    thinking=False,
    continue_final_message=False,
    tokenize=False
):
    """
    Apply a chat template to messages with optional tools and 'thinking' mode.

    Args:
        messages (list): List of role-content dicts.
        tokenizer: AutoTokenizer instance.
        tools (list): Optional tool schema.
        add_generation_prompt (bool): Whether to append a prompt for next assistant turn.
        thinking (bool): Whether to wrap/add assistant reply in <|thinking|> tags.
        continue_final_message (bool): Leave final assistant message open for generation.
        tokenize (bool): Return tokens instead of string.
    """

    formatted_messages = messages.copy()

    if thinking:
        if formatted_messages and formatted_messages[-1]["role"] == "assistant":
            # Wrap existing assistant content
            content = formatted_messages[-1]["content"]
            formatted_messages[-1]["content"] = f"<|thinking|>\n{content}\n</|thinking|>"
        else:
            # Auto-add assistant start with thinking tag
            formatted_messages.append({
                "role": "assistant",
                "content": "<|thinking|>\n"
            })
            continue_final_message = True  # ensure model continues inside <|thinking|>

    # Apply the tokenizer's chat template
    formatted_chat = tokenizer.apply_chat_template(
        formatted_messages,
        tools=tools,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message
    )

    return formatted_chat
