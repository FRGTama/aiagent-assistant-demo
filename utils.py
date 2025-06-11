def format_convo(convo):
    conversation = []
    for message in convo:
        conversation.append(f"{message.type.upper()}: {message.content}")
    return "\n".join(conversation)