import time
import openai

# Define a function that adds a delay to a Completion API call
def delayed_completion(delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    output, error = None, None
    for _ in range(max_trials):
        try:
            output = openai.ChatCompletion.create(**kwargs)
            break
        except Exception as e:
            error = e
            pass
    return output, error

def prepare_data(prompt, datum):
    postfix = "\n\nPlease respond with 'Choice 1' or 'Choice 2' without explanation."
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]
    return prompt + input_txt + postfix

def post_process(completion, choices):
    content = completion['choices'][0]['message']['content'].strip()
    result = []
    for choice in choices:
        choice_txt = "Choice" + choice
        if choice_txt in content:
            result.append(choice)
    return content, result